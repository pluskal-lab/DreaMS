#!/bin/bash
#PBS -N compute_knn_lib_dists
#PBS -l select=1:ncpus=18:mem=200gb:scratch_local=8gb
#PBS -l walltime=15:00:00

. /storage/brno2/home/romanb/msml/activate.sh
python3 /storage/brno2/home/romanb/msml/msml/inference/atlas/compute_knn_lib_dists.py
