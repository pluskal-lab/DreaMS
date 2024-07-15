# DreaMS (Deep Representations Empowering the Annotation of Mass Spectra)

Source code for the paper ["Emergence of molecular structures from repository-scale self-supervised learning on tandem mass spectra"](https://chemrxiv.org/engage/chemrxiv/article-details/6626775021291e5d1d61967f).

This GitHub repository is a work in progress. We are planning to transform it into a user-friendly Python package during July 2024.

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
from dreams.api import compute_dreams_embeddings
dreams_embeddings = compute_dreams_embeddings('data/examples/example_5_spectra.mgf')
```

The resulting `dreams_embeddings` object is a matrix with 5 rows and 1024 columns, representing 5 1024-dimensional DreaMS representations for 5 input spectra stored in the `.mgf` file.

## Work in progress
- [ ] Wrap the repository into a pip package.
- [ ] Import utilities to [matchms](https://github.com/matchms/matchms).
- [ ] DreaMS Atlas exploration demo.
- [ ] Upload weights of all models.
- [ ] Provide scripts to collect/download GeMS datasets.
- [ ] Extend `dreams.api` with more functionality (e.g. attention heads analysis).
- [ ] Add tutorial notebooks.
- [ ] Upload Murcko splits and detailed tutorial notebook.
