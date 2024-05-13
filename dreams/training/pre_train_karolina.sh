#!/bin/bash
#SBATCH --job-name pre-training
#SBATCH --account OPEN-26-5
#SBATCH --partition qgpu
#SBATCH --nodes 1
#SBATCH --gpus 8
#SBATCH --time 48:00:00

# Prepare project environment
cd "${WORK}" || exit 2
. activate.sh

# Move to running dir
cd "${EXPERIMENTS_DIR}/pre_training" || exit 3

# Export project definitions
$(python -c "from msml.definitions import export; export()")

# Run the training script
srun --export=ALL --preserve-env python3 train.py \
 --project_name SSL_VAL_4.0 \
 --job_key "${job_key}" \
 --run_name "${job_key}_V7_MoNA_1device_on_train_epoch_end_noRO" \
 --frac_masks 0.3 \
 --train_regime pre-training \
 --dataset_pth "${MONA}/mona_clean_A.pkl" \
 --val_check_interval 0.1 \
 --train_objective mask_mz_hot \
 --hot_mz_bin_size 0.05 \
 --dformat A \
 --model DreaMS \
 --ff_peak_depth 1 \
 --ff_fourier_depth 5 \
 --ff_fourier_d 512 \
 --ff_out_depth 1 \
 --prec_intens 1.1 \
 --num_devices 1 \
 --max_epochs 3000 \
 --log_every_n_steps 20 \
 --seed 3402 \
 --n_layers 7 \
 --n_heads 8 \
 --d_peak 44 \
 --d_fourier 980 \
 --lr 1e-4 \
 --batch_size 256 \
 --dropout 0.1 \
 --save_top_k 1 \
 --att_dropout 0.1 \
 --residual_dropout 0.1 \
 --ff_dropout 0.1 \
 --weight_decay 0 \
 --attn_mech dot-product \
 --train_precision 32 \
 --mask_peaks \
 --mask_intens_strategy intens_p \
 --max_peaks_n 60 \
 --ssl_probing_dataset_pths "${MERGED_DATASETS}/NIST20_MoNA_A_Murcko_split_6k_subset_3.pkl" \
 --ssl_probing_depth 0 \
 --focal_loss_gamma 5 \
 --no_transformer_bias \
 --n_warmup_steps 5000 \
 --fourier_strategy lin_float_int \
 --mz_shift_aug_p 0.2 \
 --mz_shift_aug_max 50 \
 --pre_norm \
 --graphormer_mz_diffs \
# --ret_order_loss_w 0.2 \

#--dataset_pth "${MASSIVE_DIR}"/data/msvn/msvn_A_FH1000_K10.hdf5 \

#--d_mz_token 980 \

#--d_fourier 980 \

#  --fourier_num_freqs 1024 \
# --fourier_trainable

#--dataset_pth "${MONA}/mona_clean_A.pkl" \

# --store_probing_pred
#--dataset_pth "${MASSIVE_DIR}"/data/msvn/msvn_A_FH1000_K10.hdf5 \






# --dataset_pth "${MONA}/mona_clean_A.pkl" \
# --dataset_pth "${MASSIVE_DIR}"/data/msvn/msvn_B.hdf5 \

# v2 changes
# --lr 7e-5
# --n_warmup_steps 1000
# --ret_order_loss_w 0.2
# --batch_size 256
# no --deterministic_mask
# ? --weight_decay 1e-3
# ? --n_layers 5 \

# --mask_prec

# --ssl_probing_dataset_pths "${MERGED_DATASETS}/NIST20_MoNA_A_Murcko_split_25k_subset_1.pkl,${MERGED_DATASETS}/NIST20_MoNA_A_Murcko_split_25k_subset_2.pkl,${MERGED_DATASETS}/NIST20_MoNA_A_Murcko_split_25k_subset_3.pkl"

# --mask_prec
#

# --ssl_probing_depth 0 \
# --focal_loss_gamma 0
# --mask_prec

# --dataset_pth "${MONA}"/mona_clean_A.pkl \
#--ssl_probing_dataset_pth "${MIST}/mist_fold_100_1.pkl" \
#--ssl_probing_dataset_pths "${MERGED_DATASETS}/MoNA_A_Murcko_split.pkl" \


# --n_samples 100_000 \
# --val_check_interval 5

# --spec_entropy_cleaning

# --n_warmup_steps 30_000 \


# --max_batch_var_features "lsh"

# --max_peaks_n 120
# --acc_est_weight
# --lsh_weight
# --max_batch_var_features "precursor mz" \

#
# --entropy_label_smoothing 1

#
# --no_wandb
# --n_samples 10_000_000 \