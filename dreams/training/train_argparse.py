import argparse
from pathlib import Path
from dreams.definitions import *
from dreams.utils.dformats import DataFormatBuilder


def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment setup
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--wandb_entity_name', type=str, default='mass-spec-ml')
    parser.add_argument('--job_key', type=str, required=True)
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--seed', type=int, default=1)

    # Data
    parser.add_argument('--dataset_pth', type=Path, required=True)
    parser.add_argument('--dformat', type=str, required=True, choices=['A', 'B'])
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--no_val', action='store_true')
    parser.add_argument('--val_frac', type=float, default=0.01)
    parser.add_argument('--random_fine_tuning_split', action='store_true')
    parser.add_argument('--n_samples', type=int, default=None, help='Number of dataset samples (NOTE: not random for '
                                                                    'large pre-training data).')
    parser.add_argument('--num_workers_data', type=int, default=0)
    parser.add_argument('--max_peaks_n', type=int, default=60)
    parser.add_argument('--lsh_weight', action='store_true')
    parser.add_argument('--acc_est_weight', action='store_true')
    parser.add_argument('--max_batch_var_features', type=str)
    parser.add_argument('--spec_entropy_cleaning', action='store_true')
    parser.add_argument('--mz_shift_aug_p', type=float, default=0.0)
    parser.add_argument('--mz_shift_aug_max', type=float, default=0.0)

    # Data (contrastive embedding speicific)
    parser.add_argument('--n_pos_samples', type=int, default=1)
    parser.add_argument('--n_neg_samples', type=int, default=1)
    parser.add_argument('--triplet_loss_margin', type=float, default=0.2)

    # Model (general)
    parser.add_argument('--model', type=str, default='DreaMS', choices=['DreaMS', 'VanillaBERT', 'DeepSets'])
    parser.add_argument('--pre_trained_pth', type=Path)
    parser.add_argument('--unfreeze_backbone_at_epoch', default=0, help='Either integer or False to never unfreeze.')
    parser.add_argument('--head_depth', type=int, default=1)
    parser.add_argument('--head_phi_depth', type=int, default=1, help='rho is an element-wise Deep Sets network.')

    # Model (loss)
    parser.add_argument('--train_regime', type=str, required=True, choices=['pre-training', 'fine-tuning',
        'cv-fine-tuning'])
    parser.add_argument('--hot_mz_bin_size', type=float)
    parser.add_argument('--frac_masks', type=float)
    parser.add_argument('--train_objective', type=str, required=True, choices=['mask_peak', 'mask_mz', 'mask_intensity',
        'mask_mz_hot', 'mask_peak_hot', 'shuffling', 'num_C', 'num_O', 'has_N', 'has_Cl', 'has_F', 'qed', 'fp_rdkit_2048',
        'fp_rdkit_4096', 'fp_morgan_2048', 'fp_morgan_4096', 'mol_props', 'contrastive_spec_embs'])
    parser.add_argument('--deterministic_mask', action='store_true')
    parser.add_argument('--bert801010_masking', action='store_true')
    parser.add_argument('--mask_val', type=float, default=-1)
    parser.add_argument('--mask_prec', action='store_true')
    parser.add_argument('--mask_peaks', action='store_true')
    parser.add_argument('--mask_intens_strategy', type=str, default='intens_cutoff')
    parser.add_argument('--ret_order_loss_w', type=float, default=0.0)
    parser.add_argument('--focal_loss_gamma', default=0.0, type=float, help='Gamma term for m/z masking focal loss or '
                        'classification fine-tuning tasks loss. With gamma equal to zero the loss is equivalent to '
                        'cross-entropy.')
    parser.add_argument('--focal_loss_alpha', default=None, type=float)
    parser.add_argument('--cos_reg_alpha', type=float)
    parser.add_argument('--cos_reg_reduction', type=str, choices=['max', 'mean'])

    # Model (MS-specific hyperparameters)
    parser.add_argument('--d_fourier', type=int)
    parser.add_argument('--d_peak', type=int)
    parser.add_argument('--ff_out_depth', type=int)
    parser.add_argument('--ff_peak_depth', type=int)
    parser.add_argument('--ff_fourier_depth', type=int)
    parser.add_argument('--ff_fourier_d', type=str)
    parser.add_argument('--fourier_strategy', type=str)
    parser.add_argument('--fourier_trainable', action='store_true')
    parser.add_argument('--fourier_num_freqs', type=int)
    parser.add_argument('--fourier_min_freq', type=float)
    parser.add_argument('--d_mz_token', type=int)
    parser.add_argument('--prec_intens', type=float, help='Precursor peak is prepended to spectrum with given '
                                                          'intensity.')
    parser.add_argument('--charge_feature', action='store_true')
    parser.add_argument('--graphormer_mz_diffs', action='store_true')
    parser.add_argument('--graphormer_parametrized', action='store_true')

    # Model (Transformer-specific hyperparameters)
    parser.add_argument('--vanilla_transformer', action='store_true')
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--n_heads', type=int)
    parser.add_argument('--n_layers', type=int)
    parser.add_argument('--scnorm', action='store_true')
    parser.add_argument('--attn_mech', type=str, default='dot-product',
                        choices=['dot-product', 'additive_v', 'additive_fixed'])
    parser.add_argument('--no_transformer_bias', action='store_true')
    parser.add_argument('--no_ffs_bias', action='store_true')

    # Model (training, regularization)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--max_epochs', type=int, default=3000)
    parser.add_argument('--train_precision', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--att_dropout', type=float, default=0.0)
    parser.add_argument('--residual_dropout', type=float, default=0.0)
    parser.add_argument('--ff_dropout', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--n_warmup_steps', type=int, default=0)
    parser.add_argument('--entropy_label_smoothing', type=float, default=0)

    # Validation
    parser.add_argument('--ssl_probing_dataset_pths', type=str, help='Comma-separated list of datasets paths for SSL'
                                                                     'probing.')
    parser.add_argument('--ssl_probing_depth', default=[0], help='Either int or list of ints (e.g. [0, 1, 2]).')
    parser.add_argument('--store_probing_pred', action='store_true')

    # Trainer
    parser.add_argument('--retrieval_val_pth', type=Path)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--include_val_in_train', action='store_true')
    parser.add_argument('--overfit_batches', type=int, default=0)
    parser.add_argument('--save_top_k', type=int, default=1)
    parser.add_argument('--log_every_n_steps', type=int, default=10)
    parser.add_argument('--val_check_interval', type=float, default=1.)
    parser.add_argument('--log_figs', action='store_true')

    # Infrastructure
    # parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--num_devices', type=int, default=1)

    args = parser.parse_args([] if '__file__' not in globals() else None)
    return val_prep_args(args)


def val_prep_args(args):

    # TODO: consider DeepSets

    # Validate dependencies between args

    assert args.train_regime != 'cv-fine-tuning' or args.num_devices == 1, \
        'CV fine tuning is supported only on a single device (because of wandb logging complications).'

    # bert_init = {a: vars(args)[a] for a in ['d_fourier', 'd_peak', 'nheads', 'num_layers', 'ff_out_depth',
    #     'ff_peak_depth', 'ff_fourier_depth', 'ff_fourier_d', 'fourier_strategy',
    #     'fourier_trainable', 'charge_feature']}
    # assert args.model != 'DreaMS' or args.pre_trained_pth or all(bert_init.values()), \
    #     f'Not all DreaMS hyperparameters are set: {bert_init}.'
    # assert not args.pre_trained_pth or not any(bert_init.values()), \
    #     f'Redundant DreaMS hyperparameters are set: {bert_init}.'

    # assert args.pre_trained_pth or ((args.fourier_num_freqs is not None) != (args.fourier_strategy == 'lin_float_int'))

    assert not args.pre_trained_pth or args.model == 'DreaMS' or args.model == 'VanillaBERT'

    assert not args.vanilla_transformer or (args.attn_mech == 'dot-product' and not args.graphormer_mz_diffs)

    assert not args.random_fine_tuning_split or 'fine-tuning' in args.train_regime

    # assert (args.hot_mz_bin_size is not None) == ('hot' in args.train_objective)

    assert args.frac_masks is None or 0 < args.frac_masks < 1

    assert not args.retrieval_val_pth or 'fp' in args.train_objective

    assert not args.ssl_probing_dataset_pths or args.train_regime == 'pre-training'

    assert not args.graphormer_mz_diffs or not args.d_mz_token

    assert not args.d_mz_token or not args.d_fourier

    if args.model == 'DreaMS' and not args.pre_trained_pth:
        assert sum(e for e in [args.d_fourier, args.d_peak, args.d_mz_token] if e) % args.n_heads == 0

    # assert not args.ssl_probing_depth or args.ssl_probing_dataset_pth

    # assert not args.head_depth or args.train_regime == 'fine-tuning'

    # Prepare args

    args.dformat = DataFormatBuilder(args.dformat).get_dformat()

    if args.val_check_interval and args.val_check_interval != 1. and args.val_check_interval.is_integer():
        args.val_check_interval = int(args.val_check_interval)

    if args.ff_fourier_d is not None and args.ff_fourier_d.isdigit():
        args.ff_fourier_d = int(args.ff_fourier_d)

    if args.ssl_probing_depth:
        if isinstance(args.ssl_probing_depth, int):
            args.ssl_probing_depth = [args.ssl_probing_depth]
        elif isinstance(args.ssl_probing_depth, str):
            args.ssl_probing_depth = [int(d) for d in args.ssl_probing_depth.split(',')]

    if args.ssl_probing_dataset_pths:
        args.ssl_probing_dataset_pths = [Path(p) for p in args.ssl_probing_dataset_pths.split(',')]

    if not args.d_fourier:
        args.d_fourier = 0
    if not args.d_mz_token:
        args.d_mz_token = 0

    if type(args.unfreeze_backbone_at_epoch) == bool and args.unfreeze_backbone_at_epoch:
        args.unfreeze_backbone_at_epoch = args.max_epochs + 1
    else:
        args.unfreeze_backbone_at_epoch = int(args.unfreeze_backbone_at_epoch)

    return args
