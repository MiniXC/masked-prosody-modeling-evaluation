# MPM, MPM_rand, cwt, input features, (DO MPM conv) for CONFORMER
CUDA_VISIBLE_DEVICES=2 HF_DATASETS_CACHE="/disk/scratch/swallbridge/disk_cache/temp_mpm_runs" python scripts/train_ravdess.py configs/mpm_conformer.yml --run_name ravdess_conformer_mpm --use_mpm --lr 5e-5 --overwrite

CUDA_VISIBLE_DEVICES=2 HF_DATASETS_CACHE="/disk/scratch/swallbridge/disk_cache/temp_mpm_runs" python scripts/train_ravdess.py configs/mpmrand_conformer.yml --run_name ravdess_conformer_mpmrand --use_mpm --use_mpm_random --lr 5e-5 --overwrite

CUDA_VISIBLE_DEVICES=2 HF_DATASETS_CACHE="/disk/scratch/swallbridge/disk_cache/temp_mpm_runs" python scripts/train_ravdess.py configs/cwt_conformer.yml --run_name ravdeyss_conformer_cwt --use_cwt --lr 5e-5 --overwrite

CUDA_VISIBLE_DEVICES=2 HF_DATASETS_CACHE="/disk/scratch/swallbridge/disk_cache/temp_mpm_runs" python scripts/train_ravdess.py configs/conformer.yml --run_name ravdess_conformer_input --lr 5e-5 --overwrite


# MPM, MPM_rand, cwt, input features, (DO MPM conv) for LINEAR
CUDA_VISIBLE_DEVICES=2 HF_DATASETS_CACHE="/disk/scratch/swallbridge/disk_cache/temp_mpm_runs" python scripts/train_ravdess.py configs/mpm_linear.yml --run_name ravdess_linear_mpm --use_mpm --lr 1e-3

CUDA_VISIBLE_DEVICES=2 HF_DATASETS_CACHE="/disk/scratch/swallbridge/disk_cache/temp_mpm_runs" python scripts/train_ravdess.py configs/mpmrand_linear.yml --run_name ravdess_linear_mpmrand --use_mpm --use_mpm_random --lr 1e-3

CUDA_VISIBLE_DEVICES=2 HF_DATASETS_CACHE="/disk/scratch/swallbridge/disk_cache/temp_mpm_runs" python scripts/train_ravdess.py configs/cwt_linear.yml --run_name ravdeysslinearr_cwt --use_cwt --lr 1e-3

CUDA_VISIBLE_DEVICES=2 HF_DATASETS_CACHE="/disk/scratch/swallbridge/disk_cache/temp_mpm_runs" python scripts/train_ravdess.py configs/linear.yml --run_name ravdess_linear_input --lr 1e-3



# CUDA_VISIBLE_DEVICES=0 HF_DATASETS_CACHE="/disk/scratch/swallbridge/temp_hf_cache" python scripts/train_ravdess.py configs/linear.yml --run_name ravdess_linear_input