import os
import sys
from setuptools import setup, find_packages

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Platform-specific pyopenms version (TMP workaround for macOS)
def get_pyopenms_dep():
    if sys.platform.lower().startswith("darwin"):
        return "pyopenms==3.3.0"  # macOS
    else:  # sys.platform.lower().startswith("linux"):
        return "pyopenms==3.4.0"  # Linux or other platforms

setup(
    name="dreams",
    packages=find_packages(),
    version="1.0.0",
    description="DreaMS (Deep Representations Empowering the Annotation of Mass Spectra)",
    author="DreaMS developers",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pluskal-lab/DreaMS",
    install_requires=[
        "numpy==1.25.0",
        "numba==0.58.0",
        "torch==2.2.1",
        "pytorch-lightning==2.0.8",
        "torchmetrics==1.3.2",
        "pandas==2.2.1",
        "pyarrow==15.0.2",
        "h5py==3.11.0",
        "rdkit==2023.9.6",
        "umap-learn==0.5.6",
        "seaborn==0.13.2",
        "plotly==5.20.0",
        "ase==3.22.1",
        "wandb==0.16.4",
        "pandarallel==1.6.5",
        "matchms==0.27.0",
        get_pyopenms_dep(),
        "igraph==0.11.4",
        "molplotly==1.1.7",
        "fire==0.6.0",
        "huggingface_hub==0.28.1",
        "msml @ git+https://github.com/roman-bushuiev/msml_legacy_architectures.git@main"
    ],
    extras_require={
        "dev": [
            "black==24.4.2",
            "pytest==8.2.1",
            "pytest-cov==5.0.0",
            "Cython==3.0.9",
            "SpectralEntropy @ git+https://github.com/YuanyueLi/SpectralEntropy@a1151cfcd9adc66e46f95fb3b06a660e1b0c9b56#egg=SpectralEntropy",
        ],
        "notebooks": [
            "jupyter==1.0.0",
            "ipywidgets==8.1.3",
        ],
        "search": [
            "faiss-cpu==1.9.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10.13'
)