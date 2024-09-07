<h1 align="center">DreaMS (Deep Representations Empowering the Annotation of Mass Spectra)</h1>

<!-- [![Documentation badge](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://ppiref.readthedocs.io/en/latest/?badge=latest) -->
<!-- [![Zenodo badge](https://zenodo.org/badge/DOI/10.5281/zenodo.13208732.svg)](https://doi.org/10.5281/zenodo.13208732) -->
<!-- [![Python package](https://github.com/anton-bushuiev/PPIRef/actions/workflows/python-package.yml/badge.svg)](https://github.com/anton-bushuiev/PPIRef/actions/workflows/python-package.yml) -->

<p>
  <a href="https://chemrxiv.org/engage/chemrxiv/article-details/6626775021291e5d1d61967f"><img src="https://img.shields.io/badge/ChemRxiv-10.26434-brown.svg" height="22px"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-b31b1b.svg" height="22px"></a>
  <a href="https://www.python.org/downloads/release/python-3110/"><img src="https://img.shields.io/badge/Python-3.11-blue.svg" height="22px"></a>
  <a href="https://pytorch.org/get-started/pytorch-2.0/"><img src="https://img.shields.io/badge/PyTorch-2.0.8-orange.svg" height="22px"></a>
  <a href="https://huggingface.co/datasets/roman-bushuiev/GeMS/tree/main/data"> <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg" height="22px"></a>
<p>

<p align="center">
  <img src="assets/dreams_background.png"/>
</p>

DreaMS (Deep Representations Empowering the Annotation of Mass Spectra) is a transformer-based neural network designed to interpret tandem mass spectrometry (MS/MS) data. Pre-trained in a self-supervised way on millions of unannotated spectra from our new GeMS (GNPS Experimental Mass Spectra) dataset, DreaMS acquires rich molecular representations by predicting masked spectral peaks and chromatographic retention orders. When fine-tuned for tasks such as spectral similarity, chemical properties prediction, and fluorine detection, DreaMS achieves state-of-the-art performance across various mass spectrometry interpretation tasks. The DreaMS Atlas, a comprehensive molecular network comprising 201 million MS/MS spectra annotated with DreaMS representations, along with pre-trained models and training datasets, is publicly accessible for further research and development. 🚀

This repository provides the code and tutorials to:

- 🔥 Generate **DreaMS representations** of MS/MS spectra and utilize them for downstream tasks such as spectral similarity prediction or molecular networking.
- 🤖 **Fine-tune DreaMS** for your specific tasks of interest.
- 💎 Access and utilize the extensive **GeMS dataset** of unannotated MS/MS spectra.
- 🌐 Explore the **DreaMS Atlas**, a molecular network of 201 million MS/MS spectra from diverse MS experiments annotated with DreaMS representations and metadata, such as studied species, experiment descriptions, etc.
- ⭐ Efficiently **cluster MS/MS spectra** in linear time using locality-sensitive hashing (LSH).

Additionally, for further research and development:
- 🔄 Convert conventional MS/MS data formats into our new, **ML-friendly HDF5-based format**.
- 📊 Split MS/MS datasets using **Murcko histograms** of molecular structures.

Please refer our [documentation](TODO) and a paper ["Emergence of molecular structures from repository-scale self-supervised learning on tandem mass spectra"](https://chemrxiv.org/engage/chemrxiv/article-details/6626775021291e5d1d61967f) for more details.

## Getting started


### Installation
Run the following code from the command line.

``` shell
# Download this repository
git clone https://github.com/pluskal-lab/DreaMS.git; cd DreaMS

# Create conda environment
conda create -n dreams python==3.11.0 --yes
conda activate dreams

# Install DreaMS
pip install -e .
```

If you are not familiar with conda or do not have it installed, please refer to the [official documentation](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

### Compute DreaMS representations

To compute DreaMS representations for MS/MS spectra from `.mgf` file, run the following Python code.

``` python
from dreams.api import dreams_embeddings
embs = dreams_embeddings('data/examples/example_5_spectra.mgf')
```

The resulting `embs` object is a matrix with 5 rows and 1024 columns, representing 5 1024-dimensional DreaMS representations for 5 input spectra stored in the `.mgf` file.

## Work in progress
- [ ] Wrap the repository into a pip package.
- [ ] Import utilities to [matchms](https://github.com/matchms/matchms).
- [x] DreaMS Atlas exploration demo.
- [ ] Upload weights of all models.
- [x] Provide scripts to collect/download GeMS datasets.
- [x] Extend `dreams.api` with more functionality (e.g. attention heads analysis).
- [x] Add tutorial notebooks.
- [x] Upload Murcko splits and detailed tutorial notebook.

## References

If you use DreaMS in your research, please cite the following paper:

```bibtex
@article{bushuiev2024emergence,
    author = {Bushuiev, Roman and Bushuiev, Anton and Samusevich, Raman and Brungs, Corinna and Sivic, Josef and Pluskal, Tomáš},
    title = {Emergence of molecular structures from repository-scale self-supervised learning on tandem mass spectra},
    journal = {ChemRxiv},
    doi = {doi:10.26434/chemrxiv-2023-kss3r-v2},
    year = {2024}
}
```