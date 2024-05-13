#!/bin/bash
#SBATCH --job-name dreams_fine-tuning
#SBATCH --account OPEN-26-5
#SBATCH --partition qgpu
#SBATCH --nodes 1
#SBATCH --gpus 8
#SBATCH --time 6:00:00

# Prepare project environment
# cd "/scratch/project/open-26-5" || exit 2
cd "${WORK}/msml/msml/experiments/pre_training/" || exit 2
. activate.sh
#. /scratch/project/open-26-23/antonb/miniconda3/bin/activate msml

# Move to running dir
 cd "${EXPERIMENTS_DIR}/pre_training" || exit 3

# Export project definitions
$(python -c "from msml.definitions import export; export()")

# Contrastive embedding fine-tuning
python3 train.py \
srun --export=ALL --preserve-env python3 train.py \
 --project_name CONTRASTIVE_FINE_TUNING \
 --job_key "lr5e-6_margin0.1_fixed_rel_intens_max_peaks_n100" \
 --run_name "lr5e-6_margin0.1_fixed_rel_intens_max_peaks_n100" \
 --train_objective contrastive_spec_embs \
 --train_regime fine-tuning \
 --dformat A \
 --model DreaMS \
 --lr 5e-6 \
 --batch_size 4 \
 --prec_intens 1.1 \
 --num_devices 8 \
 --max_epochs 301 \
 --log_every_n_steps 5 \
 --seed 3407 \
 --train_precision 32 \
 --val_check_interval 1.0 \
 --save_top_k -1 \
 --head_depth 0 \
 --unfreeze_backbone_at_epoch 0 \
 --dataset_pth "${MERGED_DATASETS}/MoNA_A_Murcko_split_neighbours_[M+H]+_0.05Da.pkl" \
 --pre_trained_pth "/scratch/project/open-26-5/msml/msml/experiments/pre_training/SSL_VAL_4.0/CtDh6OHlhA/epoch=6-step=71500.ckpt" \
 --n_pos_samples 1 \
 --n_neg_samples 1 \
 --triplet_loss_margin 0.1 \
 --max_peaks_n 100

# FP
#srun --export=ALL --preserve-env python3 train.py \
#  --project_name FP_1.0 \
#  --job_key "lr3e-5_bs4_sigmoid_linear" \
#  --run_name "lr3e-5_bs4_sigmoid_linear" \
#  --train_objective fp_morgan_4096 \
#  --train_regime fine-tuning \
#  --dformat A \
#  --model DreaMS \
#  --lr 3e-5 \
#  --batch_size 8 \
#  --prec_intens 1.1 \
#  --num_devices 8 \
#  --max_epochs 1001 \
#  --log_every_n_steps 5 \
#  --save_top_k -1 \
#  --seed 3407 \
#  --train_precision 32 \
#  --val_check_interval 1.0 \
#  --head_depth 1 \
#  --head_phi_depth 0 \
#  --unfreeze_backbone_at_epoch 0 \
#  --test \
#  --include_val_in_train \
#  --dataset_pth "${MIST}/mist_fold_100_0.pkl" \
#  --retrieval_val_pth "${MIST}/mist_fold_100_0_formula_isomers_mist.pkl" \
#  --pre_trained_pth "${EXPERIMENTS_DIR}/pre_training/SSL_VAL_4.0/CtDh6OHlhA/epoch=6-step=71500.ckpt" \
#  --max_peaks_n 100

# has_F
#srun --export=ALL --preserve-env python3 train.py \
#python3 train.py \
#  --project_name HAS_F_1.0 \
#  --job_key "CtDh6OHlhA/epoch=6-step=71500_v2_8bs_5e-5lr_bce" \
#  --run_name "CtDh6OHlhA/epoch=6-step=71500_v2_8bs_5e-5lr_bce" \
#  --train_objective has_F \
#  --train_regime fine-tuning \
#  --dataset_pth "${MERGED_DATASETS}/NIST20_MoNA_A_all_with_F_Murcko_split_MCE_test.pkl" \
#  --dformat A \
#  --model DreaMS \
#  --lr 5e-5 \
#  --batch_size 8 \
#  --prec_intens 1.1 \
#  --num_devices 8 \
#  --max_epochs 103 \
#  --save_top_k -1 \
#  --log_every_n_steps 5 \
#  --seed 3407 \
#  --train_precision 64 \
#  --pre_trained_pth "${EXPERIMENTS_DIR}/pre_training/SSL_VAL_4.0/CtDh6OHlhA/epoch=6-step=71500.ckpt" \
#  --val_check_interval 0.1 \
#  --max_peaks_n 100
#  --focal_loss_gamma 0.5 \
#  --focal_loss_alpha 0.8 \

# ms2prop benchmark
#python3 train.py \
#srun --export=ALL --preserve-env python3 train.py \
#  --project_name MS2PROP_1.0 \
#  --job_key "lr3e-5_bs64" \
#  --run_name "lr3e-5_bs64" \
#  --train_objective mol_props \
#  --train_regime fine-tuning \
#  --dataset_pth "${MERGED_DATASETS}/NIST20_MoNA_A_Murcko_split_MCE_test.pkl" \
#  --dformat A \
#  --model DreaMS \
#  --lr 3e-5 \
#  --batch_size 64 \
#  --prec_intens 1.1 \
#  --num_devices 8 \
#  --max_epochs 103 \
#  --log_every_n_steps 5 \
#  --head_depth 1 \
#  --seed 3407 \
#  --train_precision 64   \
#  --pre_trained_pth "${EXPERIMENTS_DIR}/pre_training/SSL_VAL_4.0/CtDh6OHlhA/epoch=6-step=71500.ckpt" \
#  --val_check_interval 0.1 \
#  --max_peaks_n 100 \
#  --save_top_k -1

# ms2prop "reimplementation"
#python3 train.py \
# srun --export=ALL --preserve-env python3 train.py \
#   --project_name MS2PROP_1.0 \
#   --job_key "ms2prop_lr3e-4_bs64" \
#   --run_name "ms2prop_lr3e-4_bs64" \
#   --train_objective mol_props \
#   --train_regime fine-tuning \
#   --dataset_pth "${MERGED_DATASETS}/NIST20_MoNA_A_Murcko_split_MCE_test.pkl" \
#   --dformat A \
#   --model DreaMS \
#   --lr 3e-4 \
#   --batch_size 64 \
#   --prec_intens 1.1 \
#   --num_devices 8 \
#   --max_epochs 103 \
#   --log_every_n_steps 5 \
#   --seed 3407 \
#   --train_precision 64 \
#   --val_check_interval 0.1 \
#   --max_peaks_n 100 \
#   --save_top_k -1 \
#   --hot_mz_bin_size 0.1 \
#   --ff_peak_depth 1 \
#   --ff_fourier_depth 5 \
#   --prec_intens 2 \
#   --n_layers 6 \
#   --n_heads 32 \
#   --d_mz_token 511 \
#   --d_peak 1 \
#   --dropout 0 \
#   --att_dropout 0 \
#   --residual_dropout 0 \
#   --ff_dropout 0 \
#   --attn_mech dot-product \
#   --head_depth 2
