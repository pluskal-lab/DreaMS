WORK IN PROGRESS

<h1 align="center">DreaMS (Deep Representations Empowering the Annotation of Mass Spectra)</h1>

<p align="center">
  <img src="assets/dreams_background.png"/>
</p>

DreaMS (Deep Representations Empowering the Annotation of Mass Spectra) is a transformer-based neural network designed to interpret tandem mass spectrometry (MS/MS) data, introduced in the paper ["Emergence of molecular structures from repository-scale self-supervised learning on tandem mass spectra"](https://chemrxiv.org/engage/chemrxiv/article-details/6626775021291e5d1d61967f). Pre-trained in a self-supervised manner on millions of unannotated spectra from the new GeMS (GNPS Experimental Mass Spectra) dataset, DreaMS learns rich molecular representations by predicting masked spectral peaks and chromatographic retention orders. Fine-tuned for tasks such as spectral similarity, molecular fingerprint prediction, chemical property inference, and fluorine detection, DreaMS achieves state-of-the-art performance across multiple mass spectrometry interpretation tasks. The DreaMS Atlas, a molecular network of 201 million MS/MS spectra annotated with DreaMS representations, and pre-trained models are publicly available for further research and development.

This repository contains the code for the DreaMS project and showcases how to:

- 🚀 Obtain DreaMS representations of MS/MS spectra and use them for downstream tasks such as spectral similarity search or molecular networking.
- 🤖 Fine-tune DreaMS for your tasks of interest.
- 💎 Download and use the large-scale GeMS dataset of unannotated MS/MS spectra.
- 🌐 Explore the DreaMS Atlas, a molecular network of 201 million MS/MS spectra annotated with DreaMS representations.

## How to install

Run the following code from the command line.

``` shell
git clone https://github.com/pluskal-lab/DreaMS.git; cd DreaMS
. scripts/install.sh
. scripts/download_models.sh
```

`git clone` command will download this GitHub repository and `install.sh` will install it. `download_models.sh` script will download pre-trained DreaMS models from [Zenodo](https://zenodo.org/records/10997887). The installation script will create a conda environment named `dreams`. If you are not familiar with conda or do not have it installed, please refer to the [official documentation](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

## How to use

To compute DreaMS representations for MS/MS spectra from `.mgf` file, run the following Python code.

``` python
from dreams.api import dreams_embeddings
dreams_embeddings = dreams_embeddings('data/examples/example_5_spectra.mgf')
```

The resulting `dreams_embeddings` object is a matrix with 5 rows and 1024 columns, representing 5 1024-dimensional DreaMS representations for 5 input spectra stored in the `.mgf` file.

## Work in progress
- [ ] Wrap the repository into a pip package.
- [ ] Import utilities to [matchms](https://github.com/matchms/matchms).
- [x] DreaMS Atlas exploration demo.
- [ ] Upload weights of all models.
- [x] Provide scripts to collect/download GeMS datasets.
- [x] Extend `dreams.api` with more functionality (e.g. attention heads analysis).
- [x] Add tutorial notebooks.
- [ ] Upload Murcko splits and detailed tutorial notebook.
