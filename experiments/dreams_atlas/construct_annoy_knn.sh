#!/bin/bash
#PBS -N construct_knn
#PBS -l select=1:ncpus=32:mem=1000gb:scratch_local=8gb
#PBS -l walltime=48:00:00

# Activate venv
module add conda-modules-py37
conda activate /storage/brno2/home/romanb/.conda/envs/tmap_env

python3 /storage/brno2/home/romanb/msml/msml/inference/atlas/construct_knn.py
