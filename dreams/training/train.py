import h5py
import pandas as pd
import pytorch_lightning as pl
import wandb
import time
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pathlib import Path
import warnings
from numba import NumbaDeprecationWarning
warnings.filterwarnings('ignore', message='.*cpp.*')  # Suppress internal torch TransfromerEncoder warnings
warnings.filterwarnings('ignore', category=NumbaDeprecationWarning)  # Supress numba warnings from UMAP
import dreams.utils.data as du
from dreams.utils.io import setup_logger
import dreams.utils.io as io
from dreams.models.dreams.dreams import DreaMS
# from dreams.models.vanilla_bert.bert import VanillaBERT
from dreams.models.heads.heads import *
from dreams.models.baselines.deep_sets import *
from dreams.training.train_argparse import parse_args
from dreams.utils.data import ContrastiveSpectraDataset
import torch
torch.set_printoptions(profile='full')
torch.set_float32_matmul_precision('high')


def main(args):

    # Prepare seeds and auxiliary variables
    seed_everything(args.seed)
    run_dir = Path(args.project_name) / args.job_key
    run_dir.mkdir(parents=True, exist_ok=True)
    args.gains_dir = run_dir / 'model_gains'
    args.gains_dir.mkdir(exist_ok=True)
    logger = setup_logger(run_dir.with_suffix('.log'))
    logger.info(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define the way to preprocess spectra (same for train or validation on independent datasets)
    spec_preproc = du.SpectrumPreprocessor(
        dformat=args.dformat, prec_intens=args.prec_intens, n_highest_peaks=args.max_peaks_n,
        spec_entropy_cleaning=args.spec_entropy_cleaning, precision=args.train_precision,
        mz_shift_aug_p=args.mz_shift_aug_p, mz_shift_aug_max=args.mz_shift_aug_max
    )

    # Define datasets
    if args.train_regime == 'pre-training':
        dataset = du.MaskedSpectraDataset(
            in_pth=args.dataset_pth, spec_preproc=spec_preproc, n_samples=args.n_samples, dformat=args.dformat,
            logger=logger, ssl_objective=args.train_objective, deterministic_mask=args.deterministic_mask,
            frac_masks=args.frac_masks, mask_prec=args.mask_prec, mask_peaks=args.mask_peaks,
            mask_intens_strategy=args.mask_intens_strategy, ret_order_pairs=args.ret_order_loss_w != 0,
            acc_est_weight=args.acc_est_weight, lsh_weight=args.lsh_weight, mask_val=args.mask_val,
            bert801010_masking=args.bert801010_masking
        )

        # with h5py.File(args.dataset_pth, 'r') as f:
        #     split_col = f['val'][:] if 'val' in f.keys() else None
        split_col = None

        if split_col is not None:
            # NOTE: max_var_features is not implemented for SplittedDataModule
            data_module = du.SplittedDataModule(
                dataset, split_mask=split_col, batch_size=args.batch_size, num_workers=args.num_workers_data,
                seed=args.seed
            )
        else:
            data_module = du.RandomSplitDataModule(
                dataset=dataset, batch_size=args.batch_size, val_frac=args.val_frac, num_workers=args.num_workers_data,
                max_var_features=dataset.data[args.max_batch_var_features] if args.max_batch_var_features else None,
            )
    elif args.train_regime in {'fine-tuning', 'cv-fine-tuning'}:
        df = pd.read_pickle(args.dataset_pth)

        if args.train_objective == 'contrastive_spec_embs':
            pos_idx_col, neg_idx_col = 'pos_idx', 'neg_idx'
            dataset = ContrastiveSpectraDataset(
                df, n_pos_samples=args.n_pos_samples, n_neg_samples=args.n_neg_samples,
                spec_preproc=spec_preproc, return_smiles=True, logger=logger,
                pos_idx_col=pos_idx_col, neg_idx_col=neg_idx_col
            )
            # Drop spectra with insufficient number of nieghbors for contrastive training
            # through modifying the split column
            mask_enough_neighbors = (
                (df[neg_idx_col].apply(len) >= args.n_neg_samples) \
                & (df[pos_idx_col].apply(len) >= args.n_pos_samples)
            )
            if logger and mask_enough_neighbors.sum() != len(df):
                n_removed = len(df) - mask_enough_neighbors.sum()
                logger.info(f'Removing {n_removed} out of {len(df)} spectra with insufficient number of neighbors.')
            assert 'fold' in df.columns or 'val' in df.columns, 'Contrastive dataset must have a split column.'
            if 'fold' in df.columns:
                df.loc[~mask_enough_neighbors, 'fold'] = 'none'
            else:
                df['fold'] = 'train'
                df.loc[df['val'], 'fold'] = 'val'
                df.loc[~mask_enough_neighbors, 'fold'] = 'none'
                del df['val']
        else:
            dataset = du.AnnotatedSpectraDataset(
                df['MSnSpectrum'].tolist(), label=args.train_objective, dformat=args.dformat, spec_preproc=spec_preproc,
                return_smiles=args.retrieval_val_pth
            )
        if args.train_regime == 'cv-fine-tuning':
            assert 'fold' in df.columns
            data_module = du.CVDataModule(
                dataset, fold_idx=df['fold'], batch_size=args.batch_size, num_workers=args.num_workers_data
            )
        elif args.random_fine_tuning_split:
            data_module = du.RandomSplitDataModule(
                dataset, val_frac=args.val_frac, batch_size=args.batch_size, num_workers=args.num_workers_data
            )
        else:
            split_col = 'val' if 'val' in df.columns else 'fold'
            assert split_col in df.columns
            data_module = du.SplittedDataModule(
                dataset, split_mask=df[split_col], batch_size=args.batch_size, num_workers=args.num_workers_data,
                n_train_samples=args.n_samples, seed=args.seed, include_val_in_train=args.include_val_in_train
            )

    # Log dataset sizes
    cv = True if isinstance(data_module, du.CVDataModule) else False
    # if not cv:
    #     n_train_samples, n_train_batches = len(data_module.train_dataloader().dataset), len(data_module.train_dataloader())
    #     n_val_samples, n_val_batches = len(data_module.val_dataloader().dataset), len(data_module.val_dataloader())
    # logger.info(f'# train samples: {n_train_samples} ({n_train_batches} batches)')
    # logger.info(f'# val samples: {n_val_samples} ({n_val_batches} batches)')

    # If cross validation, iterate over folds
    for i in range(data_module.get_num_folds() if cv else 1):
        if cv:
            data_module.setup_fold_index(i)

        # Define model
        if args.model == 'DreaMS':
            if not args.pre_trained_pth:
            #     model = DreaMS.load_from_checkpoint(args.pre_trained_pth, map_location=torch.device(device))
            # else:
                model = DreaMS(args, spec_preproc)

        # elif args.model == 'VanillaBERT':
        #     if args.pre_trained_pth:
        #         model = VanillaBERT.load_from_checkpoint(args.pre_trained_pth)
        #     else:
        #         model = VanillaBERT(
        #             gains_dir=args.gains_dir,
        #             d_fourier=args.d_fourier,
        #             d_peak=args.d_peak,
        #             nheads=args.n_heads,
        #             num_layers=args.n_layers,
        #             ff_peak_depth=args.ff_peak_depth,
        #             ff_fourier_depth=args.ff_fourier_depth,
        #             ff_out_depth=args.ff_out_depth,
        #             dropout=args.dropout,
        #             lr=args.lr,
        #             fourier_strategy=args.fourier_strategy,
        #             weight_decay=args.weight_decay,
        #             charge_feature=args.charge_feature,
        #             ssl_objective=args.train_objective,
        #             dformat=args.dformat,
        #             fourier_trainable=args.fourier_trainable,
        #             fourier_num_freqs=args.fourier_num_freqs,
        #             ff_fourier_d=args.ff_fourier_d,
        #             hot_mz_bin_size=args.hot_mz_bin_size,
        #             n_warmup_steps=args.n_warmup_steps,
        #             fourier_min_freq=args.fourier_min_freq,
        #             # batch_size=args.batch_size
        #         )
        elif args.model == 'DeepSets':
            if args.train_objective.startswith('fp'):
                model = DeepSetsPeaksFingerprint(args.train_objective, lr=args.lr)
            elif args.train_objective in {'num_C', 'num_O'}:
                model = DeepSetsPeakIntReg(lr=args.lr)
            elif args.train_objective in {'qed', 'ms2prop_labels'}:
                model = DeepSetsPeakReg(lr=args.lr, out_dim=10 if args.train_objective == 'ms2prop_labels' else 1)
            elif args.train_objective in {'has_N', 'has_Cl', 'has_F'}:
                model = DeepSetsPeakBinCls(lr=args.lr)
        else:
            NotImplementedError(f'Model {args.model} is not implemented')

        # Append fine-tuning heads
        if 'fine-tuning' in args.train_regime and args.model != 'DeepSets':
            backbone = args.pre_trained_pth if args.pre_trained_pth else DreaMS(args, spec_preproc)
            if args.train_objective in {'fp_morgan_2048', 'fp_morgan_4096', 'fp_rdkit_2048', 'fp_rdkit_4096'}:
                model = FingerprintHead(backbone=backbone, fp_str=args.train_objective,
                                        lr=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
                                        retrieval_val_pth=args.retrieval_val_pth, batch_size=args.batch_size,
                                        unfreeze_backbone_at_epoch=args.unfreeze_backbone_at_epoch,
                                        store_val_out_dir=run_dir / f'val_out_{args.dataset_pth.stem}',
                                        head_depth=args.head_depth, head_phi_depth=args.head_phi_depth)
            # TODO: refactor backbone
            if args.train_objective in {'num_C', 'num_O'}:
                model = IntRegressionHead(args.pre_trained_pth, args.lr, args.weight_decay)
            elif args.train_objective in {'qed'}:
                model = RegressionHead(args.pre_trained_pth, args.lr, args.weight_decay, sigmoid=True)
            elif args.train_objective in {'has_N', 'has_Cl', 'has_F'}:
                model = BinClassificationHead(args.pre_trained_pth, args.lr, args.weight_decay,
                                              focal_loss_alpha=args.focal_loss_alpha,
                                              focal_loss_gamma=args.focal_loss_gamma)
            elif args.train_objective == 'contrastive_spec_embs':
                # df_smiles_similarities = pd.read_pickle(MERGED_DATASETS / 'nist20_clean_MoNA_contrastive_v2_10ppm_smiles_similarities_asymmetric.pkl')#io.append_to_stem(args.dataset_pth, f'smiles_similarities'))
                model = ContrastiveHead(args.pre_trained_pth, args.lr, args.weight_decay,
                                        triplet_loss_margin=args.triplet_loss_margin)
            elif args.train_objective == 'mol_props':
                mol_props_calc = dataset.prop_calc
                model = RegressionHead(backbone, args.lr, args.weight_decay, sigmoid=False, out_dim=len(mol_props_calc),
                                       mol_props_calc=mol_props_calc, head_depth=args.head_depth, dropout=args.dropout)

        # Set float64 weights
        if args.train_precision == 64:
            model = model.double()

        # Define wandb log
        if not args.no_wandb:
            if cv:
                wandb.init(reinit=True, project=args.project_name, name=f'{args.run_name} [fold_{i}]', config=args,
                    group=args.run_name, entity=args.wandb_entity_name)
                wandb_logger = WandbLogger()
            else:
                wandb_logger = WandbLogger(project=args.project_name, name=args.run_name, config=args, 
                    entity=args.wandb_entity_name)
        else:
            wandb_logger = None

        # Define trainer callbacks (TODO: understand the behavior of find_unused_parameters)
        callbacks = [
            LearningRateMonitor(logging_interval='step'),
            pl.callbacks.ModelCheckpoint(
                monitor='Train loss', save_top_k=args.save_top_k, mode='min',
                dirpath=run_dir, save_last=True, every_n_train_steps=1000
            )
        ]

        if args.train_regime == 'pre-training':

            # Define SSL probling callback
            if args.ssl_probing_dataset_pths:
                for pth in args.ssl_probing_dataset_pths:
                    df = pd.read_pickle(pth)
                    probing_data_module = du.SplittedDataModule(
                        du.AnnotatedSpectraDataset(
                            df['MSnSpectrum'].tolist(),
                            label='fp_maccs_166',
                            spec_preproc=spec_preproc,
                            dformat=args.dformat,
                            return_smiles=args.store_probing_pred
                        ),
                        split_mask=df['val'] if 'val' in df.columns else df['fold'],
                        batch_size=64
                    )
                    callbacks.append(
                        du.SSLProbingValidation(probing_data_module, n_hidden_layers=args.ssl_probing_depth, prefix=pth.stem,
                                                save_fps_dir=args.gains_dir / 'probing' if args.store_probing_pred else None)
                    )

        # Define trainer
        strategy = pl.strategies.DDPStrategy(
            find_unused_parameters=False if args.train_regime == 'pre-training' else True
        ) if not cv else None
        trainer = pl.Trainer(
            strategy=strategy, max_epochs=args.max_epochs, logger=wandb_logger if not args.no_wandb else None,
            accelerator=device, devices=args.num_devices, log_every_n_steps=args.log_every_n_steps,
            precision=args.train_precision, overfit_batches=args.overfit_batches, callbacks=callbacks,
            num_sanity_val_steps=0, use_distributed_sampler=args.num_devices > 1,
            val_check_interval=None if args.no_val else args.val_check_interval,
            limit_val_batches=0 if args.no_val else None
        )

        if not args.no_wandb and trainer.global_rank == 0:

            # Watch model on the main process
            wandb_logger.watch(model, log_graph=False)
            wandb_logger.experiment.config.update({
                    'num_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
            })

        # Compute validation metrics before the training
        if args.train_regime == 'pre-training' and not args.no_val:
            trainer.validate(model, data_module)

        trainer.validate(model, dataloaders=[l for l in [data_module.val_dataloader()] if l is not None])

        trainer.fit(model, train_dataloaders=data_module.train_dataloader(),
                    val_dataloaders=[l for l in [data_module.val_dataloader()] if l is not None])


if __name__ == '__main__':
    args = parse_args()
    main(args)
