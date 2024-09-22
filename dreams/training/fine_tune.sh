#!/bin/bash
#SBATCH --job-name DreaMS_fine-tuning
#SBATCH --account OPEN-29-57
#SBATCH --partition qgpu
#SBATCH --nodes 1
#SBATCH --gpus 8
#SBATCH --time 10:00:00

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate dreams

# Export project definitions
$(python -c "from dreams.definitions import export; export()")

# Move to running dir
cd "${DREAMS_DIR}" || exit 3

# Run the training script
# Replace `python3 training/train.py` with `srun --export=ALL --preserve-env python3 training/train.py \`
# when executing on a SLURM cluster via `sbatch`.
python3 training/train.py \
 --project_name MolecularProperties \
 --job_key "my_run_name" \
 --run_name "my_run_name" \
 --train_objective mol_props \
 --train_regime fine-tuning \
 --dataset_pth "${DATA_DIR}/MassSpecGym_MurckoHist_split.hdf5" \
 --dformat A \
 --model DreaMS \
 --lr 3e-5 \
 --batch_size 64 \
 --prec_intens 1.1 \
 --num_devices 8 \
 --max_epochs 103 \
 --log_every_n_steps 5 \
 --head_depth 1 \
 --seed 3407 \
 --train_precision 64   \
 --pre_trained_pth "${PRETRAINED}/ssl_model.ckpt" \
 --val_check_interval 0.1 \
 --max_peaks_n 100 \
 --save_top_k -1

# Contrastive fine-tuning
# python3 training/train.py \
#  --project_name CONTRASTIVE_FINE_TUNING \
#  --job_key "lr5e-6_margin0.1_fixed_rel_intens_max_peaks_n100" \
#  --run_name "lr5e-6_margin0.1_fixed_rel_intens_max_peaks_n100" \
#  --train_objective contrastive_spec_embs \
#  --train_regime fine-tuning \
#  --dformat A \
#  --model DreaMS \
#  --lr 5e-6 \
#  --batch_size 4 \
#  --prec_intens 1.1 \
#  --num_devices 8 \
#  --max_epochs 301 \
#  --log_every_n_steps 5 \
#  --seed 3407 \
#  --train_precision 32 \
#  --val_check_interval 1.0 \
#  --save_top_k -1 \
#  --head_depth 0 \
#  --unfreeze_backbone_at_epoch 0 \
#  --dataset_pth "${MERGED_DATASETS}/MoNA_A_Murcko_split_neighbours_[M+H]+_0.05Da.pkl" \
#  --pre_trained_pth "${PRETRAINED}/ssl_model.ckpt" \
#  --n_pos_samples 1 \
#  --n_neg_samples 1 \
#  --triplet_loss_margin 0.1 \
#  --max_peaks_n 100
