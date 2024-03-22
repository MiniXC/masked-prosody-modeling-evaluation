# MPM, MPM_rand, cwt, input features, (DO MPM conv) for CONFORMER
CUDA_VISIBLE_DEVICES=1 TIMIT_PATH="/disk/scratch1/swallbridge/datasets/timit" HF_DATASETS_CACHE="/disk/scratch/swallbridge/disk_cache/temp_mpm_runs" python scripts/train_timit.py configs/mpm_conformer.yml --run_name timit_conformer_mpm --use_mpm --lr 5e-4 --overwrite

CUDA_VISIBLE_DEVICES=1 TIMIT_PATH="/disk/scratch1/swallbridge/datasets/timit" HF_DATASETS_CACHE="/disk/scratch/swallbridge/disk_cache/temp_mpm_runs" python scripts/train_timit.py configs/mpmrand_conformer.yml --run_name timit_conformer_mpmrand --use_mpm --use_mpm_random --lr 5e-4 --overwrite

CUDA_VISIBLE_DEVICES=1 TIMIT_PATH="/disk/scratch1/swallbridge/datasets/timit" HF_DATASETS_CACHE="/disk/scratch/swallbridge/disk_cache/temp_mpm_runs" python scripts/train_timit.py configs/cwt_conformer.yml --run_name timit_conformer_cwt --use_cwt --lr 5e-4 --overwrite

CUDA_VISIBLE_DEVICES=1 TIMIT_PATH="/disk/scratch1/swallbridge/datasets/timit" HF_DATASETS_CACHE="/disk/scratch/swallbridge/disk_cache/temp_mpm_runs" python scripts/train_timit.py configs/conformer.yml --run_name timit_conformer_input --lr 5e-4 --overwrite


# MPM, MPM_rand, cwt, input features, (DO MPM conv) for LINEAR
CUDA_VISIBLE_DEVICES=1 TIMIT_PATH="/disk/scratch1/swallbridge/datasets/timit" HF_DATASETS_CACHE="/disk/scratch/swallbridge/disk_cache/temp_mpm_runs" python scripts/train_timit.py configs/mpm_linear.yml --run_name timit_linear_mpm --use_mpm --lr 1.0e-3

CUDA_VISIBLE_DEVICES=1 TIMIT_PATH="/disk/scratch1/swallbridge/datasets/timit" HF_DATASETS_CACHE="/disk/scratch/swallbridge/disk_cache/temp_mpm_runs" python scripts/train_timit.py configs/mpmrand_linear.yml --run_name timit_linear_mpmrand --use_mpm --use_mpm_random --lr 1.0e-3

CUDA_VISIBLE_DEVICES=1 TIMIT_PATH="/disk/scratch1/swallbridge/datasets/timit" HF_DATASETS_CACHE="/disk/scratch/swallbridge/disk_cache/temp_mpm_runs" python scripts/train_timit.py configs/cwt_linear.yml --run_name timit_linear_cwt --use_cwt --lr 1.0e-3

CUDA_VISIBLE_DEVICES=1 TIMIT_PATH="/disk/scratch1/swallbridge/datasets/timit" HF_DATASETS_CACHE="/disk/scratch/swallbridge/disk_cache/temp_mpm_runs" python scripts/train_timit.py configs/linear.yml --run_name timit_linear_input --lr 1.0e-3



# CUDA_VISIBLE_DEVICES=0 HF_DATASETS_CACHE="/disk/scratch/swallbridge/temp_hf_cache" python scripts/train_ravdess.py configs/linear.yml --run_name ravdess_linear_input