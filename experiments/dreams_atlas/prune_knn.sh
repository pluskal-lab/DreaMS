#!/bin/bash
#PBS -N prune_knn
#PBS -l select=1:ncpus=8:mem=600gb:scratch_local=8gb
#PBS -l walltime=5:00:00

. /storage/brno2/home/romanb/msml/activate.sh
python3 /storage/brno2/home/romanb/msml/msml/inference/atlas/prune_knn.py
