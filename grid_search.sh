accelerate launch scripts/train_burn.py configs/baseline.yml --run_name burn_linear_baseline
accelerate launch scripts/train_ravdess.py configs/baseline.yml --run_name ravdess_linear_baseline
accelerate launch scripts/train_timit.py configs/baseline.yml --run_name timit_linear_baseline
# grid search over  bin size and mask length
# - bin sizes: 16, 32, 64, 128, 512, 1024
# - mask lengths: 1, 2, 4, 8, 16, 32, 64, 128
accelerate launch scripts/train_burn.py configs/mpm.yml --run_name burn_gs_bin16_mask1 --bin_size 16 --mask_length 1
