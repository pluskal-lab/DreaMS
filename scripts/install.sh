#!/bin/bash
# This script install the conda encironment for DreaMS and should be run from the root of the DreaMS repository

# Create a conda environment
conda create -n dreams python==3.11.0 --yes
conda activate dreams

# Install dependencies
pip install torch==2.2.1
pip install pytorch-lightning==2.2.1
pip install pandas==2.2.1
pip install pyarrow==15.0.2
pip install h5py==3.11.0
pip install rdkit==2023.9.5
pip install umap-learn==0.5.6
pip install seaborn==0.13.2
pip install plotly==5.20.0
pip install ase==3.22.1
pip install wandb==0.16.4
pip install pandarallel==1.6.5
pip install matchms==0.24.2
pip install pyopenms==3.0.0
pip install igraph==0.11.4
pip install Cython==3.0.9
pip install git+https://github.com/YuanyueLi/SpectralEntropy.git@master
pip install jupyter==1.0.0
pip install molplotly==1.1.7

# Install DreaMS
pip install -e .

# Install legacy DreaMS architectures to load pre-trained models (will be removed in the future)
pip install git+https://github.com/roman-bushuiev/msml_legacy_architectures.git@main
