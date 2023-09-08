# bin sizes: 4, 8, 16, 32, 64, 128, 512, 1024
# mask sizes: 1, 2, 4, 8, 16, 32, 64, 128

# check if --gpu argument is 0
if [ "$1" == "0" ]; then
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/mpm_linear.yml --run_name burn_bin4_mask1_linear --mpm_bin_size 4 --mpm_mask_size 1 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/mpm_conformer.yml --run_name burn_bin4_mask1_conformer --mpm_bin_size 4 --mpm_mask_size 1 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/mpm_linear.yml --run_name burn_bin4_mask2_linear --mpm_bin_size 4 --mpm_mask_size 2 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/mpm_conformer.yml --run_name burn_bin4_mask2_conformer --mpm_bin_size 4 --mpm_mask_size 2 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/mpm_linear.yml --run_name burn_bin4_mask4_linear --mpm_bin_size 4 --mpm_mask_size 4 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/mpm_conformer.yml --run_name burn_bin4_mask4_conformer --mpm_bin_size 4 --mpm_mask_size 4 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/mpm_linear.yml --run_name burn_bin4_mask8_linear --mpm_bin_size 4 --mpm_mask_size 8 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/mpm_conformer.yml --run_name burn_bin4_mask8_conformer --mpm_bin_size 4 --mpm_mask_size 8 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/mpm_linear.yml --run_name burn_bin4_mask16_linear --mpm_bin_size 4 --mpm_mask_size 16 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/mpm_conformer.yml --run_name burn_bin4_mask16_conformer --mpm_bin_size 4 --mpm_mask_size 16 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/mpm_linear.yml --run_name burn_bin4_mask32_linear --mpm_bin_size 4 --mpm_mask_size 32 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/mpm_conformer.yml --run_name burn_bin4_mask32_conformer --mpm_bin_size 4 --mpm_mask_size 32 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/mpm_linear.yml --run_name burn_bin4_mask64_linear --mpm_bin_size 4 --mpm_mask_size 64 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/mpm_conformer.yml --run_name burn_bin4_mask64_conformer --mpm_bin_size 4 --mpm_mask_size 64 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/mpm_linear.yml --run_name burn_bin4_mask128_linear --mpm_bin_size 4 --mpm_mask_size 128 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/mpm_conformer.yml --run_name burn_bin4_mask128_conformer --mpm_bin_size 4 --mpm_mask_size 128 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/linear.yml --run_name burn_bin8_mask1_linear --mpm_bin_size 8 --mpm_mask_size 1 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/conformer.yml --run_name burn_bin8_mask1_conformer --mpm_bin_size 8 --mpm_mask_size 1 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/linear.yml --run_name burn_bin8_mask2_linear --mpm_bin_size 8 --mpm_mask_size 2 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/conformer.yml --run_name burn_bin8_mask2_conformer --mpm_bin_size 8 --mpm_mask_size 2 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/linear.yml --run_name burn_bin8_mask4_linear --mpm_bin_size 8 --mpm_mask_size 4 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/conformer.yml --run_name burn_bin8_mask4_conformer --mpm_bin_size 8 --mpm_mask_size 4 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/linear.yml --run_name burn_bin8_mask8_linear --mpm_bin_size 8 --mpm_mask_size 8 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/conformer.yml --run_name burn_bin8_mask8_conformer --mpm_bin_size 8 --mpm_mask_size 8 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/linear.yml --run_name burn_bin8_mask16_linear --mpm_bin_size 8 --mpm_mask_size 16 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/conformer.yml --run_name burn_bin8_mask16_conformer --mpm_bin_size 8 --mpm_mask_size 16 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/linear.yml --run_name burn_bin8_mask32_linear --mpm_bin_size 8 --mpm_mask_size 32 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/conformer.yml --run_name burn_bin8_mask32_conformer --mpm_bin_size 8 --mpm_mask_size 32 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/linear.yml --run_name burn_bin8_mask64_linear --mpm_bin_size 8 --mpm_mask_size 64 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/conformer.yml --run_name burn_bin8_mask64_conformer --mpm_bin_size 8 --mpm_mask_size 64 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/linear.yml --run_name burn_bin8_mask128_linear --mpm_bin_size 8 --mpm_mask_size 128 --use_mpm --mpm_layer 7 --wandb_mode=online
    CUDA_VISIBLE_DEVICES=$1 python scripts/train_$2.py configs/conformer.yml --run_name burn_bin8_mask128_conformer --mpm_bin_size 8 --mpm_mask_size 128 --use_mpm --mpm_layer 7 --wandb_mode=online
fi
if [ "$1" == "1" ]; then
    CUDA_VISIBLE_DEVICES=$1 BURN_PATH="/disk/scratch/s1764494/data/bu_radio_$1" TIMIT_PATH="/disk/scratch/s1764494/data/timit_$1" python scripts/train_$2.py configs/mpm_linear.yml --run_name burn_bin16_mask1_linear --mpm_bin_size 16 --mpm_mask_size 1 --use_mpm --mpm_layer 7 --wandb_mode=online
    