import os
from setuptools import setup, find_packages

# Get the long description from the README file
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="dreams",
    packages=find_packages(),
    version="1.0.0",
    description="DreaMS (Deep Representations Empowering the Annotation of Mass Spectra)",
    author="DreaMS developers",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pluskal-lab/MassSpecGym",
    install_requires=[
        "numpy",
        "numba",
        "torch",
        "pytorch-lightning",
        "torchmetrics",
        "pandas",
        "pyarrow",
        "h5py",
        "rdkit",
        "umap-learn",
        "seaborn",
        "plotly",
        "ase",
        "wandb",
        "pandarallel",
        "matchms",
        "pyopenms",
        "igraph",
        "molplotly",
        "fire",
        "huggingface_hub",
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
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)
