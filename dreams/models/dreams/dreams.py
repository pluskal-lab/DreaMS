import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from torchmetrics.functional import pairwise_cosine_similarity
from typing import Union, Dict
from torch.utils.data.dataloader import DataLoader
from dreams.models.layers.fourier_features import FourierFeatures
from dreams.models.layers.feed_forward import FeedForward
import dreams.utils.spectra as su
import dreams.utils.data as du
from dreams.definitions import NIST20, MONA
from dreams.models.optimization.schedulers import NoamScheduler
from dreams.models.dreams.layers import TransformerEncoder
from dreams.models.optimization.losses_metrics import FocalLoss


class DreaMS(pl.LightningModule):

    def __init__(self, args, spec_preproc: du.SpectrumPreprocessor):
        """
        Argument and their dependencies are validated in dreams/experiments/pre_training/train_arparse.py.
        """

        super(DreaMS, self).__init__()
        self.save_hyperparameters()

        # TODO: move namespace args to constructor args?
        self.spec_preproc = spec_preproc
        self.gains_dir = args.gains_dir
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.train_objective = args.train_objective
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.charge_feature = args.charge_feature
        self.d_fourier = args.d_fourier
        self.d_peak = args.d_peak
        self.d_mz_token = args.d_mz_token
        self.d_model = sum(d for d in [self.d_fourier, self.d_peak, self.d_mz_token] if d)
        args.d_model = self.d_model
        self.dformat = args.dformat
        self.hot_mz_bin_size = args.hot_mz_bin_size
        self.n_warmup_steps = args.n_warmup_steps
        self.vanilla_transformer = args.vanilla_transformer
        self.batch_size = args.batch_size
        self.log_figs = args.log_figs
        self.entropy_label_smoothing = args.entropy_label_smoothing
        self.graphormer_mz_diffs = args.graphormer_mz_diffs
        self.graphormer_parametrized = args.graphormer_parametrized
        self.fourier_strategy = args.fourier_strategy
        self.ret_order_loss_w = args.ret_order_loss_w
        self.cos_reg_alpha = args.cos_reg_alpha
        self.cos_reg_reduction = args.cos_reg_reduction
        self.mask_val = args.mask_val
        if self.graphormer_mz_diffs and self.graphormer_parametrized:
            args.d_graphormer_params = args.d_fourier if self.d_fourier else 1
        else:
            args.d_graphormer_params = 0

        token_dim = 2
        if args.charge_feature:
            token_dim += 1

        # Fourier features encoding (for m/z's only)
        if self.d_fourier:
            self.fourier_enc = FourierFeatures(strategy=args.fourier_strategy, num_freqs=args.fourier_num_freqs,
                                           x_min=args.dformat.max_tbxic_stdev if not args.fourier_min_freq else args.fourier_min_freq,
                                           x_max=args.dformat.max_mz,
                                           trainable=args.fourier_trainable)

            self.ff_fourier = FeedForward(in_dim=self.fourier_enc.num_features(), out_dim=args.d_fourier, dropout=args.dropout,
                                      depth=args.ff_fourier_depth, hidden_dim=args.ff_fourier_d, bias=not args.no_ffs_bias)
        # Tokenized input m/z values
        elif self.d_mz_token:
            self.mz_tokenizer = nn.Embedding(
                # +1 for masking token
                num_embeddings=1 + su.num_hot_classes(max_val=args.dformat.max_mz, bin_size=args.hot_mz_bin_size),
                embedding_dim=self.d_mz_token,
                padding_idx=0
            )
            self.ff_mz_token = FeedForward(in_dim=self.d_mz_token, hidden_dim=args.d_mz_token, out_dim=args.d_mz_token, depth=2, dropout=args.dropout)

        # Input position-wise feed forward
        # (batch_size, peaks_n, token_dim) -> (batch_size, peaks_n, d_peak)
        self.ff_peak = FeedForward(in_dim=token_dim, hidden_dim=args.d_peak, out_dim=args.d_peak, depth=args.ff_peak_depth,
                                   dropout=args.dropout, bias=not args.no_ffs_bias)

        # Stack of the Transformer encoder layers (i.e. BERT)
        # (batch_size, peaks_n, d_model) -> (batch_size, peaks_n, d_model)
        if args.vanilla_transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, dim_feedforward=self.d_model * 4,
                                                       nhead=self.n_heads, activation='gelu', dropout=args.dropout,
                                                       batch_first=True, norm_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        else:
            self.transformer_encoder = TransformerEncoder(args)

        # Output position-wise feed forward
        if self.train_objective.endswith('hot'):
            # (batch_size, peaks_n, d_model) -> (batch_size, peaks_n, num_one_hot_bins)
            self.ff_out = FeedForward(in_dim=self.d_model, hidden_dim=self.d_model, depth=args.ff_out_depth, act_last=False,
                                      out_dim=su.num_hot_classes(max_val=args.dformat.max_mz, bin_size=args.hot_mz_bin_size),
                                      dropout=args.dropout, bias=True)
            # self.ff_out = nn.Sequential(
            #     nn.Linear(self.d_model, 256, bias=False),
            #     # nn.Dropout(0.5),
            #     nn.Linear(256, su.num_hot_classes(max_val=args.dformat.max_mz, bin_size=args.hot_mz_bin_size), bias=False),
            # )
            if self.train_objective == 'mask_peak_hot':
                self.ff_out_intens = FeedForward(in_dim=self.d_model, hidden_dim=self.d_model, depth=args.ff_out_depth,
                                                 act_last=False, out_dim=su.num_hot_classes(max_val=1.0, bin_size=0.05),
                                                 dropout=args.dropout, bias=False)

            # Define m/z masking loss function
            self.mz_masking_loss = FocalLoss(gamma=args.focal_loss_gamma, return_softmax_out=True)

        # elif self.train_objective.startswith('mask'):
        elif self.train_objective == 'mask_mz':
            # (batch_size, peaks_n, d_model) -> (batch_size, peaks_n, 2 (m/z, intensity))
            self.ff_out = FeedForward(in_dim=self.d_model, hidden_dim=self.d_model, out_dim=1, depth=args.ff_out_depth,
                                      act_last=False, dropout=args.dropout, bias=False)
            self.mz_masking_loss = nn.MSELoss()
        elif self.train_objective.startswith('shuffling'):
            # (batch_size, peaks_n, d_model) -> (batch_size, peaks_n, 1 (shuffled/non-shuffled label))
            self.ff_out = FeedForward(in_dim=self.d_model, hidden_dim=self.d_model, out_dim=1, depth=args.ff_out_depth,
                                      act_last=False, dropout=args.dropout, bias=False)
        else:
            self.ff_out = None

        # Linear projection for the binary classification of RO from a pair of spectra embeddings
        if self.ret_order_loss_w:
            self.ro_out = nn.Linear(2 * self.d_model, 1, bias=False)

    def forward(self, spec, charge=None):
        """ Returns embeddings from the last Transformer encoder layer. """

        # Generate padding mask
        padding_mask = spec[:, :, 0] == 0

        # Append charge to each token
        if self.charge_feature:
            if charge is None:
                raise ValueError
            charge_features = ~padding_mask * charge.unsqueeze(-1)
            spec = torch.cat([spec, charge_features.unsqueeze(-1)], dim=-1)

        # Lift peaks to d_peak (m/z's are normalized)
        peak_embs = self.ff_peak(self.__normalize_spec(spec))

        # ms2prop variant
        # peak_embs = spec[:, :, 1].unsqueeze(-1)

        # Concatenate with fourier features (d_peak -> d_peak + d_fourier ("num_fourier_features" -> d_fourier))
        if self.d_fourier:
            fourier_features = self.ff_fourier(self.fourier_enc(spec[..., [0]]))
            spec = torch.cat([peak_embs, fourier_features], dim=-1)
        elif self.d_mz_token:
            tokenized_mzs = self.mz_tokenizer(
                su.to_classes(spec[..., [0]], max_val=self.dformat.max_mz, bin_size=self.hot_mz_bin_size,
                              special_vals=[self.mask_val]).squeeze()
            )
            tokenized_mzs = self.ff_mz_token(tokenized_mzs)
            spec = torch.cat([peak_embs, tokenized_mzs], dim=-1)
        else:
            spec = peak_embs

        graphormer_dists = None
        if self.graphormer_mz_diffs:
            if self.d_fourier:
                graphormer_dists = fourier_features.unsqueeze(2) - fourier_features.unsqueeze(1)
            else:
                graphormer_dists = spec[..., 0].unsqueeze(2) - spec[..., 0].unsqueeze(1)
                graphormer_dists = graphormer_dists.unsqueeze(-1)

        # Transformer encoder blocks
        if self.vanilla_transformer:
            spec = self.transformer_encoder(spec, src_key_padding_mask=padding_mask)
        else:
            spec = self.transformer_encoder(spec, padding_mask, graphormer_dists)

        return spec

    def spec_ssl_step(self, spec_mask, spec_real, mask, charge):

        if self.train_objective.startswith('mask'):

            # Select only predicted and real values corresponding to masked tokens
            # NOTE: [mask] performs (bs, n, d) -> (bs * n mask True bits, d) reshaping
            # NOTE: [mask] is applied to predictions later in-place to keep pred_embs as a return value
            pred_embs = self(spec_mask, charge)
            real = spec_real[mask]

            if self.train_objective.endswith('hot'):

                # Decode peak embeddings to one hot m/z classes
                pred_mz = self.ff_out(pred_embs[mask])

                # Convert ground-truth m/z values to one-hot classes
                real_mz = su.to_hot(real[..., [0]], max_val=self.dformat.max_mz, bin_size=self.hot_mz_bin_size)

                # Compute loss
                loss, p_mz = self.mz_masking_loss(pred_mz, real_mz)

                # Prediction of intensity hot classes
                if self.train_objective == 'mask_peak_hot':

                    # Convert ground-truth intensities to one-hot classes
                    real_intens = su.to_hot(real[..., [1]], max_val=1.0, bin_size=0.05)

                    # Decode peak embeddings to one hot intensity classes
                    pred_intens = self.ff_out_intens(pred_embs[mask])

                    # Compute loss
                    loss += 0.5 * F.cross_entropy(pred_intens, real_intens, reduction='none')

                # Entropy label smoothing
                if self.entropy_label_smoothing > 0:
                    loss -= self.entropy_label_smoothing * torch.distributions.Categorical(p_mz).entropy().mean()
            elif self.train_objective == 'mask_mz':
                pred_mz = self.ff_out(pred_embs[mask])
                real_mz = real[..., [0]]
                loss = self.mz_masking_loss(pred_mz, real_mz)
            else:
                raise NotImplementedError('Not tested with the updated boolean masking.')
                # Prediction of m/z, intensity or both continuous values
                # real = self.__normalize_spec(real)
                # pred, real = pred[..., 0], real[..., 0]
                # if self.train_objective == 'mask_mz':
                #     pred, real = pred[..., 0], real[..., 0]
                # elif self.train_objective == 'mask_intensity':
                #     pred, real = pred[..., 1], real[..., 1]
                # loss = torch.sqrt(F.mse_loss(pred, real))
        elif self.train_objective.startswith('shuffling'):
            raise NotImplementedError('Not tested with the updated boolean masking.')
            # Predict shuffling indicator from the precursor peak token
            # prec_peak_mask = torch.zeros((spec_mask.shape[0], 1), dtype=torch.long)
            # pred = self(spec_mask, charge=charge, mask_i=prec_peak_mask).squeeze()
            # real = spec_real
            # loss = F.binary_cross_entropy_with_logits(pred, real)
        else:
            raise NotImplementedError

        return loss, pred_embs, pred_mz, real_mz

    def step(self, data, batch_idx, log_prefix):

        # Retention order prediction from two spectra along with masking objectives
        if self.ret_order_loss_w:

            # Masking SSL for 1st spectrum
            loss1, embs1, pred_mz1, real_mz1 = self.spec_ssl_step(
                data['spec_mask_1'], data['spec_real_1'], data['mask_1'],
                data['charge_1'] if 'charge_1' in data.keys() else None
            )
            if 'spec_weight_1' in data.keys():
                loss1 = self.__weight_loss(loss1, data['spec_weight_1'], peak_mask=data['mask_1'])
            else:
                loss1 = loss1.sum() / loss1.numel()

            self.log(
                f'{log_prefix} spec1 accuracy',
                (torch.argmax(pred_mz1, dim=-1) == torch.argmax(real_mz1, dim=-1)).sum() / sum(real_mz1.shape[:-1]),
                sync_dist=True
            )

            # Masking SSL for 2nd spectrum
            loss2, embs2, pred_mz2, real_mz2 = self.spec_ssl_step(
                data['spec_mask_2'], data['spec_real_2'], data['mask_2'],
                data['charge_2'] if 'charge_2' in data.keys() else None
            )
            if 'spec_weight_2' in data.keys():
                loss2 = self.__weight_loss(loss1, data['spec_weight_2'], peak_mask=data['mask_2'])
            else:
                loss2 = loss2.sum() / loss2.numel()

            # Retention order SSL from concatenated precursor embeddings
            prec_embs12 = torch.cat([embs1[:, 0, :], embs2[:, 0, :]], dim=-1)
            loss_ro = F.binary_cross_entropy(
                F.sigmoid(self.ro_out(prec_embs12)).squeeze(),
                data['ro_label']
            )

            # Sum losses
            loss_mask = (loss1 + loss2) / 2
            loss = (1 - self.ret_order_loss_w) * loss_mask + self.ret_order_loss_w * loss_ro
            embs = torch.stack([embs1, embs2])

            # Log losses
            self.log_dict({
                f'{log_prefix} spec1 loss': loss1,
                f'{log_prefix} spec2 loss': loss2,
                f'{log_prefix} RO loss': loss_ro
            }, sync_dist=True)

        # Masking SSL prediction for a single spectrum
        else:
            loss, embs, pred_mz, real_mz = self.spec_ssl_step(
                data['spec_mask'], data['spec_real'], data['mask'],
                data['charge'] if 'charge' in data.keys() else None
            )
            if 'spec_weight' in data.keys():
                loss = self.__weight_loss(loss, data['spec_weight'], peak_mask=data['mask'])

            self.log(
                f'{log_prefix} accuracy',
                (torch.argmax(pred_mz, dim=-1) == torch.argmax(real_mz, dim=-1)).sum() / sum(real_mz.shape[:-1]),
                sync_dist=True
            )

        loss = loss.sum() / loss.numel()

        if self.cos_reg_alpha and self.cos_reg_reduction:
            cos_reg = pairwise_cosine_similarity(embs[:, 0, :], zero_diagonal=True)
            if self.cos_reg_reduction == 'mean':
                cos_reg = cos_reg.mean()
            elif self.cos_reg_reduction == 'max':
                cos_reg = cos_reg.max()
            else:
                raise ValueError
            cos_reg = self.cos_reg_alpha * cos_reg
            self.log(f'{log_prefix} masking loss', loss, sync_dist=True)
            self.log(f'{log_prefix} cos regularization loss', cos_reg, sync_dist=True)
            loss += cos_reg

        self.log(f'{log_prefix} loss', loss, sync_dist=True)
        return loss, embs

    def training_step(self, data, batch_idx):
        return self.step(data, batch_idx, 'Train')[0]

    def validation_step(self, data, batch_idx):
        return self.step(data, batch_idx, 'Val')[0]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.n_warmup_steps and self.n_warmup_steps > 0:
            lr_scheduler = {
                'scheduler': NoamScheduler(optimizer, self.n_warmup_steps),
                'interval': 'step',
                'frequency': 1,
            }
            return [optimizer], [lr_scheduler]
        return optimizer

    def __weight_loss(self, loss, spec_weights, peak_mask=None, reduce=True):

        # Replicate each spectrum weight to get peak weights: (bs) -> (bs * n mask True bits)
        # ((bs, n) * (bs, 1))[(bs, n)] -> (bs * n mask True bits)
        if peak_mask is not None:
            spec_weights = (peak_mask * spec_weights[:, None])[peak_mask]

        # Rescale weights to sum up to 1
        spec_weights /= spec_weights.sum()

        # Weight the loss
        loss *= spec_weights

        # Sum reduce (affine combination with the previous operation)
        if reduce:
            loss = loss.sum()

        return loss

    def __normalize_spec(self, spec):
        """
        Normalizes raw m/z values. Notice, that it is not in dataset `__getitem__ `because raw m/z values are still needed
        for Fourier features. Intensities are supposed to be normalized in `__getitem__`.
        """
        return spec / torch.tensor([self.dformat.max_mz, 1.], device=self.device, dtype=self.dtype)

    # def get_attention_scores(self, data):
    #     attn_scores = {}
    #     def get_attn_scores(name):
    #         def hook(model, input, output):
    #             attn_scores[name] = output[1]
    #         return hook
    #
    #     for i in range(self.n_layers):
    #         self.transformer_encoder.atts[i].register_forward_hook(get_attn_scores(i))
    #     _ = self(data['spec'], data['charge'])
    #
    #     return attn_scores

    # def on_train_epoch_end(self):
    #
    #     # Attention entropy validation
    #     if isinstance(self.logger, pl.loggers.WandbLogger):
    #         val = du.AttentionEntropyValidation(
    #             NIST20 / 'nist20_clean_A.pkl', dformat=self.dformat, spec_preproc=self.spec_preproc, n_samples=300,
    #             as_plot=True, save_out_basename=self.gains_dir / f'attn_entropy_e{self.current_epoch}'
    #         )
    #         val.set_model_gains(self.get_attention_scores(val.get_data(self.device, torch_dtype=self.dtype)))
    #         res = val.get_res()
    #         self.trainer.logger.experiment.log({'Attention entropy': res})

    def on_train_epoch_end(self):
        # TODO: refactor each validation to a separate callback similarly to linear probing

        # Spectrum -> InChI key retrieval validation
        val = du.SpecRetrievalValidation(
            NIST20 / 'nist20_clean_spec_entropy_[M+H]+_retrieval.pkl',
            NIST20 / 'nist20_clean_spec_entropy_[M+H]+_50k_pairs_retrieval.pkl',
            dformat=self.dformat,
            spec_preproc=self.spec_preproc
        )
        data = val.get_data(device=self.device, torch_dtype=self.dtype)
        embeddings = self.get_embeddings(data)
        val.set_model_gains(embeddings)
        self.log_dict({f'[NIST20] {k}': v for k, v in val.get_res().items()}, sync_dist=True)

        # Correlation validation
        for df_pth in MONA.glob('*pairs.pkl'):
            val = du.CorrelationValidation(MONA / 'mona_clean_A_full.pkl', df_pth, dformat=self.dformat,
                                           spec_preproc=self.spec_preproc)

            data = val.get_data(device=self.device, torch_dtype=self.dtype)
            embeddings = self.get_embeddings(data)
            val.set_model_gains(embeddings)
            self.log_dict({f'[MoNA] {k}': v for k, v in val.get_res().items()}, sync_dist=True)

        for df_pth in NIST20.glob('*pairs.pkl'):
            val = du.CorrelationValidation(NIST20 / 'nist20_clean_A.pkl', df_pth, dformat=self.dformat,
                                           spec_preproc=self.spec_preproc)

            data = val.get_data(device=self.device, torch_dtype=self.dtype)
            embeddings = self.get_embeddings(data)
            val.set_model_gains(embeddings)
            self.log_dict({f'[NIST20] {k}': v for k, v in val.get_res().items()}, sync_dist=True)

        # Contrastive validation
        # for df_pth in NIST20.glob('*groups.pkl'):
        #     val = du.ContrastiveValidation(
        #         NIST20 / 'nist20_clean_A.pkl', df_pth, dformat=self.dformat, n_instances=7, n_samples=100,
        #         save_out_basename=self.gains_dir / f'contrastive_{df_pth.stem}_e{self.current_epoch}',
        #         spec_preproc=self.spec_preproc
        #     )
        #
        #     data = val.get_data(device=self.device, torch_dtype=self.dtype)
        #     embeddings = self.get_embeddings(data)
        #     val.set_model_gains(embeddings)
        #     self.log_dict(val.get_res(), sync_dist=True)
        #     if self.log_figs and isinstance(self.logger, pl.loggers.WandbLogger):
        #         self.trainer.logger.experiment.log({val.get_name(): val.get_umap_plot()})

        # kNN validation
        for df_pth in NIST20.glob('*groups.pkl'):
            val = du.KNNValidation(
                NIST20 / 'nist20_clean_A.pkl', df_pth, dformat=self.dformat, n_instances=7, n_samples=100, k=[1, 5],
                save_out_basename=self.gains_dir / f'contrastive_{df_pth.stem}_e{self.current_epoch}',
                spec_preproc=self.spec_preproc
            )

            data = val.get_data(device=self.device, torch_dtype=self.dtype)
            embeddings = self.get_embeddings(data)
            val.set_model_gains(embeddings)
            self.log_dict(val.get_res(), sync_dist=True)

        # # Attention entropy validation
        # if isinstance(self.logger, pl.loggers.WandbLogger):
        #     val = AttentionEntropyValidation(
        #         NIST20 / 'nist20_clean_A.pkl', dformat=self.dformat, n_samples=300, as_plot=True,
        #         save_out_basename=self.gains_dir / f'attn_entropy_e{self.current_epoch}'
        #     )
        #     val.set_model_gains(self.get_attention_scores(val.get_data(self.device, torch_dtype=self.dtype)))
        #     res = val.get_res()
        #     self.trainer.logger.experiment.log({'Attention entropy': res})
    def get_embeddings(self, data, tqdm_batches=False, batch_size=None):
        return get_embeddings(self, data, batch_size=self.batch_size if batch_size is None else batch_size,
                              tqdm_batches=tqdm_batches)


def get_embeddings(model: DreaMS, data: Dict, batch_size=None, tqdm_batches=False, precursor_only=True,
                   layers_idx=None):
    """
    TODO: replace with equivalent utils.dreams method.

    Performs a forward pass of the model on `data` samples and extracts the embeddings from the last Transformer
    encoder layer.
    :param batch_size: If not None, performs forward pass in batches of `batch_size` samples to minimize memory
                       utilization.
    :param tqdm_batches: If True, shows progress bar over batches.
    :param precursor_only: If True, returns only the embedding of the precursor "master" token.
    :param layers_idx: Indices of Transformer encoder layers to extract embeddings from. If None, only the last layer
                      is considered.
    """
    model.eval()

    spec = data['spec'].to(device=model.device, dtype=model.dtype)
    charge = data['charge'].to(device=model.device, dtype=model.dtype)

    if not batch_size:
        batch_size = spec.shape[0]
    if not layers_idx:
        layers_idx = [model.n_layers - 1]

    # Define hooks extracting intermediate representations
    embeddings = {}
    def get_embeddings_hook(name):
        def hook(model, input, output):
            embs = output.detach()
            if precursor_only:
                embs = embs[:, 0, :]
            if name not in embeddings.keys():
                embeddings[name] = embs
            else:
                embeddings[name] = torch.cat([embeddings[name], embs])
        return hook

    # Register hooks
    hook_handles = [model.transformer_encoder.ffs[i].register_forward_hook(get_embeddings_hook(i)) for i in layers_idx]

    # Infer embeddings for batches of input to minimize memory utilization
    idx_batches = list(range(0, spec.shape[0], batch_size))
    for i in tqdm(idx_batches, desc='Computing DreaMS', disable=not tqdm_batches):
        with torch.inference_mode():
            _ = model(spec[i:i+batch_size], None)

    # Remove hooks
    for h in hook_handles:
        h.remove()

    if len(layers_idx) == 1:
        return embeddings[layers_idx[0]]
    return embeddings


# def precursor_embeddings_from_pkl(model_pth: Path, in_pth: Path, batch_size=32, dformat='A'):
#     """
#     TODO: Generalize to different file formats and embedding types.
#     TODO: Parallelize?
#     """
#
#     # model = DreaMS.load_from_checkpoint(PRETRAINED / 'AzwizvV5Q7_epoch=0-step=1500.ckpt', map_location=torch.device('cpu')).double().eval()
#     # model = DreaMS.load_from_checkpoint(PRETRAINED / '771dJv5GN8_step89k.ckpt', map_location=torch.device('cpu')).double().eval()
#
#     if in_pth.suffix != '.pkl':
#         raise ValueError('Currently only ".pkl" format is supported to extract embeddings.')
#
#     model = DreaMS.load_from_checkpoint(model_pth)
#     df = pd.read_pickle(in_pth)
#     dformat = dformats.DataFormatBuilder(dformat).get_dformat()
#
#     val = du.ManualValidation(dformats.to_format(df, filter=False, dformat=dformat), dformat)
#     data = val.get_data(device=model.device, torch_dtype=model.dtype)
#     embs = get_embeddings(model, data, batch_size=batch_size, tqdm_batches=True, layers_idx=None)
#     torch.save(embs, io.append_to_stem(io.append_to_stem(in_pth, 'embs'), model_pth.stem).with_suffix('.pt'))
