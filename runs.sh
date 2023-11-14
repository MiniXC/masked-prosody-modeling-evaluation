# for runs on just one GPU/CPU use the following
python scrists/train_ravdess.py configs/conformer.yml --run_name ravdess_conformer_raw_algo --use_algorithmic_features --use_cwt

# only use for multiple GPUs/TPUs
accelerate launch scrists/train_ravdess.py configs/conformer.yml --run_name ravdess_conformer_raw_algo --use_algorithmic_features --use_cwt
accelerate launch scripts/train_timit.py configs/conformer.yml --run_name timit_conformer_raw_algo --use_algorithmic_features --use_cwt
accelerate launch scripts/train_burn.py configs/conformer.yml --run_name burn_conformer_raw_algo --use_algorithmic_features --use_cwt


python scripts/train_burn.py configs/conformer.yml --run_name ravdess_conformer_raw_algo --use_algorithmic_features --use_cwt --overwrite_data --wandb_mode online

python scripts/train_burn.py configs/mpm_conformer.yml --run_name burn_conformer_mpm --overwrite_data
python scripts/train_ravdess.py configs/mpm_conformer.yml --run_name ravdess_conformer_mpm --overwrite_data
python scripts/train_timit.py configs/mpm_conformer.yml --run_name timit_conformer_mpm --overwrite_data

# layer eval
# accelerate launch scripts/train_burn.py configs/linear.yml --run_name burn_layer_0 --mpm_bin_size 16 --mpm_mask_size 8 --use_mpm --mpm_layer 0 --num_workers 8
# accelerate launch scripts/train_burn.py configs/linear.yml --run_name burn_layer_1 --mpm_bin_size 16 --mpm_mask_size 8 --use_mpm --mpm_layer 1 --num_workers 8
# accelerate launch scripts/train_burn.py configs/linear.yml --run_name burn_layer_2 --mpm_bin_size 16 --mpm_mask_size 8 --use_mpm --mpm_layer 2 --num_workers 8
# accelerate launch scripts/train_burn.py configs/linear.yml --run_name burn_layer_4 --mpm_bin_size 16 --mpm_mask_size 8 --use_mpm --mpm_layer 4 --num_workers 8
# accelerate launch scripts/train_burn.py configs/linear.yml --run_name burn_layer_6 --mpm_bin_size 16 --mpm_mask_size 8 --use_mpm --mpm_layer 6 #--num_workers 8
# accelerate launch scripts/train_burn.py configs/linear.yml --run_name burn_layer_8 --mpm_bin_size 16 --mpm_mask_size 8 --use_mpm --mpm_layer 8 #--num_workers 8
# accelerate launch scripts/train_burn.py configs/linear.yml --run_name burn_layer_10 --mpm_bin_size 16 --mpm_mask_size 8 --use_mpm --mpm_layer 10 #--num_workers 8
# accelerate launch scripts/train_burn.py configs/linear.yml --run_name burn_layer_12 --mpm_bin_size 16 --mpm_mask_size 8 --use_mpm --mpm_layer 12 #--num_workers 8
# accelerate launch scripts/train_burn.py configs/linear.yml --run_name burn_layer_14 --mpm_bin_size 16 --mpm_mask_size 8 --use_mpm --mpm_layer 14 #--num_workers 8
# accelerate launch scripts/train_burn.py configs/linear.yml --run_name burn_layer_15 --mpm_bin_size 16 --mpm_mask_size 8 --use_mpm --mpm_layer 15 #--num_workers 8

# accelerate launch scripts/train_burn.py configs/conformer.yml --run_name burn_local_conformer_mpm_8 --mpm_bin_size 16 --mpm_mask_size 8 --use_mpm --mpm_layer 7 --num_workers 8
# accelerate launch scripts/train_burn.py configs/conformer.yml --run_name burn_conformer_raw --num_workers 8
# accelerate launch scripts/train_burn.py configs/conformer.yml --run_name burn_global_conformer_mpm_8 --mpm_bin_size 16 --mpm_mask_size 128 --use_mpm --mpm_layer 7 --num_workers 8
# accelerate launch scripts/train_burn.py configs/linear.yml --run_name burn_local_linear_mpm_8 --mpm_bin_size 16 --mpm_mask_size 8 --use_mpm --mpm_layer 7  --num_workers 8
# accelerate launch scripts/train_burn.py configs/linear.yml --run_name burn_linear_raw --num_workers 8
# accelerate launch scripts/train_burn.py configs/linear.yml --run_name burn_global_linear_mpm_8 --mpm_bin_size 16 --mpm_mask_size 128 --use_mpm --mpm_layer 7 --num_workers 8
# accelerate launch scripts/train_ravdess.py configs/conformer.yml --run_name ravdess_local_conformer_mpm_8 --mpm_bin_size 16 --mpm_mask_size 8 --use_mpm --mpm_layer 7 --num_workers 8
# accelerate launch scripts/train_ravdess.py configs/conformer.yml --run_name ravdess_conformer_raw --num_workers 8
# accelerate launch scripts/train_ravdess.py configs/conformer.yml --run_name ravdess_global_conformer_mpm_8 --mpm_bin_size 16 --mpm_mask_size 128 --use_mpm --mpm_layer 7 --num_workers 8
# accelerate launch scripts/train_ravdess.py configs/linear.yml --run_name ravdess_local_linear_mpm_8 --mpm_bin_size 16 --mpm_mask_size 8 --use_mpm --mpm_layer 7 --num_workers 8
# accelerate launch scripts/train_ravdess.py configs/linear.yml --run_name ravdess_linear_raw --num_workers 8
# accelerate launch scripts/train_ravdess.py configs/linear.yml --run_name ravdess_global_linear_mpm_8 --mpm_bin_size 16 --mpm_mask_size 128 --use_mpm --mpm_layer 7 --num_workers 8
# accelerate launch scripts/train_timit.py configs/conformer.yml --run_name timit_local_conformer_mpm_8 --mpm_bin_size 16 --mpm_mask_size 8 --use_mpm --mpm_layer 7 --num_workers 8
# accelerate launch scripts/train_timit.py configs/conformer.yml --run_name timit_conformer_raw --num_workers 8
# accelerate launch scripts/train_timit.py configs/conformer.yml --run_name timit_global_conformer_mpm_8 --mpm_bin_size 16 --mpm_mask_size 128 --use_mpm --mpm_layer 7 --num_workers 8
# accelerate launch scripts/train_timit.py configs/linear.yml --run_name timit_local_linear_mpm_8 --mpm_bin_size 16 --mpm_mask_size 8 --use_mpm --mpm_layer 7 --num_workers 8
# accelerate launch scripts/train_timit.py configs/linear.yml --run_name timit_linear_raw --num_workers 8
# accelerate launch scripts/train_timit.py configs/linear.yml --run_name timit_global_linear_mpm_8 --mpm_bin_size 16 --mpm_mask_size 128 --use_mpm --mpm_layer 7 --num_workers 8
