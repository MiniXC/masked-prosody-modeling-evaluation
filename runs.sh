# accelerate launch scripts/train_burn.py configs/lstm.yml --run_name burn_local_lstm --mpm_bin_size 16 --mpm_mask_size 8
# accelerate launch scripts/train_ravdess.py configs/lstm.yml --run_name ravdess_local_lstm --mpm_bin_size 16 --mpm_mask_size 8
accelerate launch scripts/train_timit.py configs/lstm.yml --run_name timit_local_lstm --mpm_bin_size 16 --mpm_mask_size 8