import h5py
import torch
import pandas as pd
import click
from tqdm import tqdm
from pathlib import Path
from typing import Union, Type
from torch.utils.data.dataloader import DataLoader
from dataclasses import dataclass
import dreams.utils.data as du
import dreams.utils.io as io
import dreams.utils.dformats as dformats
from dreams.models.dreams.dreams import DreaMS as DreaMSModel
from dreams.models.heads.heads import *
from dreams.definitions import *


class PreTrainedModel:
    def __init__(self, model: Union[DreaMSModel, FineTuningHead], n_highest_peaks: int = 100):
        self.model = model.eval()
        self.n_highest_peaks = n_highest_peaks

    @classmethod
    def from_ckpt(cls, ckpt_path: Path, ckpt_cls: Union[Type[DreaMSModel], Type[FineTuningHead]], n_highest_peaks: int):
        return cls(
            ckpt_cls.load_from_checkpoint(
                ckpt_path,
                backbone_pth=PRETRAINED / 'ssl_model.ckpt',
                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ),
            n_highest_peaks=n_highest_peaks
        )

    @classmethod
    def from_name(cls, name: str):
        if name == DREAMS_EMBEDDING:
            ckpt_path = PRETRAINED / 'embedding_model.ckpt'
            ckpt_cls = ContrastiveHead
            n_highest_peaks = 100
        # elif name == 'Fluorine probability':
        #     ckpt_path = EXPERIMENTS_DIR / 'pre_training/HAS_F_1.0/CtDh6OHlhA/epoch=6-step=71500_v2_16bs_5e-5lr_gamma0.5_alpha0.8/epoch=30-step=111000.ckpt'
        #     ckpt_cls = BinClassificationHead
        #     n_highest_peaks = 100
        # elif name == 'Molecular properties':
        #     ckpt_path = EXPERIMENTS_DIR / f'pre_training/MS2PROP_1.0/lr3e-5_bs64/epoch=4-step=4000.ckpt'
        #     ckpt_cls = RegressionHead
        #     n_highest_peaks = 100
        else:
            # TODO: Include all pre-trained models
            raise ValueError(f'{name} is not a valid pre-trained model name. Choose from: {cls.available_models()}')

        return cls.from_ckpt(ckpt_path, ckpt_cls, n_highest_peaks)

    # def __init_model(self, model: Union[DreaMSModel, FineTuningHead], n_highest_peaks: int = 100):
    #     self.model = model.eval()
    #     self.n_highest_peaks = n_highest_peaks

    @staticmethod
    def available_models():
        return ['Fluorine probability', 'Molecular properties', DREAMS_EMBEDDING]


def compute_dreams_predictions(
        model_ckpt: Union[PreTrainedModel, FineTuningHead, DreaMSModel, Path, str], spectra: Union[Path, str],
        model_cls=None, batch_size=32, tqdm_batches=True, write_log=False, n_highest_peaks=None, title='',
        **msdata_kwargs
    ):

    # TODO: samples in tqmd progress instead of batches

    # Load pre-trained model
    if not isinstance(model_ckpt, PreTrainedModel):
        if isinstance(model_ckpt, str):
            if '/' in model_ckpt:
                model_ckpt = Path(model_ckpt)
            else:
                title = model_ckpt
        if isinstance(model_ckpt, str):
            model_ckpt = PreTrainedModel.from_name(model_ckpt)
        elif isinstance(model_ckpt, Path):
            model_ckpt = PreTrainedModel.from_ckpt(model_ckpt, model_cls, n_highest_peaks)
        else:
            model_ckpt = PreTrainedModel(model_ckpt, n_highest_peaks)

    # Initialize spectrum preprocessing
    spec_preproc = du.SpectrumPreprocessor(dformat=dformats.DataFormatA(), n_highest_peaks=model_ckpt.n_highest_peaks)

    # Load a dataset of spectra
    if isinstance(spectra, str):
        spectra = Path(spectra)
    spectra_pth = spectra

    msdata = du.MSData.load(spectra, **msdata_kwargs)
    spectra = msdata.to_torch_dataset(spec_preproc)
    dataloader = DataLoader(spectra, batch_size=batch_size, shuffle=False, drop_last=False)

    logger = io.setup_logger(spectra_pth.with_suffix('.log'))
    tqdm_logger = io.TqdmToLogger(logger)

    # Compute predictions
    model = model_ckpt.model
    # TODO: consider model name
    if not title:
        title = 'DreaMS_prediction'
    preds = None
    for i, batch in enumerate(tqdm(
        dataloader,
        desc='Computing ' + title.replace('_', ' '),
        disable=not tqdm_batches, file=tqdm_logger if write_log else None
    )):
        with torch.inference_mode():
            pred = model(batch['spec'].to(device=model.device, dtype=model.dtype))

            # Store predictions to cpu to avoid high memory allocation issues
            pred = pred.cpu()
            if preds is None:
                preds = pred
            else:
                preds = torch.cat([preds, pred])

    preds = preds.squeeze().cpu().numpy()

    # TODO: move to outer scope
    # msdata.add_column(title, preds)
    return preds


def compute_dreams_embeddings(pth, batch_size=32, tqdm_batches=True, write_log=False, **msdata_kwargs):
    return compute_dreams_predictions(
        DREAMS_EMBEDDING, pth, batch_size=batch_size, tqdm_batches=tqdm_batches, write_log=write_log, **msdata_kwargs
    )


def generate_all_dreams_predictions(pth: Union[Path, str], batch_size=32, tqdm_batches=True,
                                 spec_col='PARSED PEAKS', prec_mz_col='PRECURSOR M/Z'):

    if isinstance(pth, str):
        pth = Path(pth)

    out_pth = io.append_to_stem(pth, 'DreaMS').with_suffix('.hdf5')

    with h5py.File(out_pth, 'w') as f:
        for m in PreTrainedModel.available_models():
            preds = compute_dreams_predictions(m, pth, batch_size=batch_size, tqdm_batches=tqdm_batches,
                                           spec_col=spec_col, prec_mz_col=prec_mz_col)
            preds = preds.squeeze().cpu().numpy()

            if m == 'Molecular properties':
                for i, p in enumerate(mu.MolPropertyCalculator().prop_names):
                    f.create_dataset(p, data=preds[:, i])
            else:
                f.create_dataset(m, data=preds)


# TODO: refactor after `get_dreams_predictions` is refactored
# def get_dreams_embeddings(model: Union[Path, str, DreaMS], df_spectra: Union[Path, str, pd.DataFrame], layers_idx=None,
#                           precursor_only=True, batch_size=32, tqdm_batches=True, spec_col='PARSED PEAKS',
#                           prec_mz_col='PRECURSOR M/Z', n_highest_peaks=128, return_attention_matrices=False,
#                           spec_preproc: du.SpectrumPreprocessor = None):
#
#     # Load model and spectra
#     model = load_model(model, model_cls=DreaMS)
#     dataloader = load_spectra(df_spectra, batch_size, spec_col=spec_col, prec_mz_col=prec_mz_col,
#                               n_highest_peaks=n_highest_peaks, spec_preproc=spec_preproc)
#
#     # Determine layers to extract embeddings from
#     if not layers_idx:
#         layers_idx = [model.n_layers - 1]
#
#     # Register hooks extracting embeddings
#     hook_handles = []
#     embeddings = {}
#     def get_embeddings_hook(name):
#         def hook(model, input, output):
#             embs = output.detach()
#             if precursor_only:
#                 embs = embs[:, 0, :]
#             if name not in embeddings.keys():
#                 embeddings[name] = embs
#             else:
#                 embeddings[name] = torch.cat([embeddings[name], embs])
#         return hook
#     for i in layers_idx:
#         hook_handles.append(model.transformer_encoder.ffs[i].register_forward_hook(get_embeddings_hook(i)))
#
#     # Register hooks extracting attention matrices
#     attn_matrices = {}
#     if return_attention_matrices:
#         def get_attn_scores_hook(name):
#             def hook(model, input, output):
#                 output = output[1].detach()
#                 if name not in attn_matrices.keys():
#                     attn_matrices[name] = output
#                 else:
#                     attn_matrices[name] = torch.cat([attn_matrices[name], output])
#             return hook
#         for i in layers_idx:
#             hook_handles.append(model.transformer_encoder.atts[i].register_forward_hook(get_attn_scores_hook(i)))
#
#     # Perform forward passes
#     for batch in tqdm(dataloader, desc='Computing DreaMS', disable=not tqdm_batches):
#         with torch.inference_mode():
#             _ = model(batch['spec'].to(device=model.device, dtype=model.dtype))
#
#     # Remove hooks
#     for h in hook_handles:
#         h.remove()
#
#     # Simplify output if only one layer was requested (no 1-element list)
#     if len(layers_idx) == 1:
#         embeddings = embeddings[layers_idx[0]]
#
#     if return_attention_matrices:
#         return embeddings, attn_matrices
#     return embeddings


# TODO: get_dreams_embeddings as a wrapper over pre-defined get_dreams_predictions
# TODO: cache loaded models in class


@click.command()
@click.option('-i', '--input_pth', help="Path to MS/MS spectra.", type=Path)
@click.option('--batch_size', help="Batch size.", type=int, default=32)
def main(input_pth, batch_size):
    calc_dreams_embeddings(pth=input_pth, batch_size=batch_size)


if __name__ == '__main__':
    main()
