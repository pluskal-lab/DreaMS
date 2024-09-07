<h1 align="center">DreaMS (Deep Representations Empowering the Annotation of Mass Spectra)</h1>

<p align="center">
  <img src="assets/dreams_background.png"/>
</p>

DreaMS (Deep Representations Empowering the Annotation of Mass Spectra) is a transformer-based neural network designed to interpret tandem mass spectrometry (MS/MS) data. Pre-trained in a self-supervised way on millions of unannotated spectra from our new GeMS (GNPS Experimental Mass Spectra) dataset, DreaMS acquires rich molecular representations by predicting masked spectral peaks and chromatographic retention orders. When fine-tuned for tasks such as spectral similarity, molecular fingerprint prediction, chemical property inference, and fluorine detection, DreaMS achieves state-of-the-art performance across various mass spectrometry interpretation tasks. The DreaMS Atlas, a comprehensive molecular network comprising 201 million MS/MS spectra annotated with DreaMS representations, along with pre-trained models and training datasets, is publicly accessible for further research and development. 🚀

This repository provides the code and tutorials to:

- 🔥 Generate DreaMS representations of MS/MS spectra and utilize them for downstream tasks such as spectral similarity prediction or molecular networking.
- 🤖 Fine-tune DreaMS for your specific tasks of interest.
- 💎 Access and utilize the extensive GeMS dataset of unannotated MS/MS spectra.
- 🌐 Explore the DreaMS Atlas, a molecular network of 201 million MS/MS spectra from diverse MS experiments annotated with DreaMS representations and metadata, such as studied species, experiment descriptions, etc.

Additionally, for machine learning research:
- 🔄 Convert conventional MS/MS data formats into our new, ML-friendly HDF5-based format.
- 📊 Split MS/MS datasets using Murcko histograms of molecular structures.
- ⭐ Efficiently cluster MS/MS spectra using locality-sensitive hashing (LSH) in linear time.

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
- [x] Upload Murcko splits and detailed tutorial notebook.
