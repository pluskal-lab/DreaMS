#!/bin/bash
#PBS -N diameter
#PBS -l select=1:ncpus=8:mem=200gb:scratch_local=8gb
#PBS -l walltime=24:00:00

. /storage/brno2/home/romanb/msml/activate.sh
python3 /storage/brno2/home/romanb/msml/msml/inference/atlas/compute_knn_diameter.py
