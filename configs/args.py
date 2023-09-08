from dataclasses import dataclass


@dataclass
class TrainingArgs:
    lr: float = 1e-4
    lr_schedule: str = "linear_with_warmup"
    lr_warmup_steps: int = 1000
    gradient_clip_val: float = 1.0
    timit_phon_focal_loss_alpha: float = 0.5
    timit_word_focal_loss_alpha: float = 0.5
    burn_focal_loss_alpha: float = 0.5
    burn_focal_loss_gamma: float = 2.0
    checkpoint_path: str = "checkpoints"
    output_path: str = "outputs"
    run_name: str = None
    wandb_mode: str = "offline"
    wandb_project: str = None
    wandb_dir: str = "wandb"
    n_steps: int = 2000
    batch_size: int = 32
    seed: int = 0
    num_workers: int = None
    burn_dataset: str = "cdminix/bu_radio"
    burn_train_split: str = "train[:90%]"
    burn_val_split: str = "train[90%:]"
    ravdess_dataset: str = "narad/ravdess"
    ravdess_train_split: str = "train[:90%]"
    ravdess_val_split: str = "train[90%:]"
    timit_dataset: str = "timit_asr"
    timit_train_split: str = "train[:50%]"
    timit_val_split: str = "test[:10%]"
    log_every_n_steps: int = 10
    do_full_eval: bool = True
    do_save: bool = False
    save_onnx: bool = False
    eval_only: bool = False
    eval_every_n_steps: int = 100
    save_every_n_steps: int = 1000
    push_to_hub: bool = False
    hub_repo: str = None
    drop_last: bool = False
    overwrite_data: bool = False
    use_mpm: bool = False
    mpm_bin_size: str = "128"
    mpm_mask_size: str = "16"
    mpm_checkpoint_step: int = 10000


@dataclass
class BURNCollatorArgs:
    max_words: int = 256
    name: str = "default_burn"
    vocex: str = "cdminix/vocex"
    vocex_fp16: bool = False
    mpm: str = None
    mpm_layer: int = -1
    pitch_min: float = 50
    pitch_max: float = 400
    energy_min: float = 0
    energy_max: float = 1
    vad_min: float = 0
    vad_max: float = 1
    use_algorithmic_features: bool = False


@dataclass
class RAVDESSCollatorArgs:
    max_frames: int = 512
    vocex: str = "cdminix/vocex"
    vocex_fp16: bool = False
    name: str = "default_ravdess"
    mpm: str = None
    mpm_layer: int = -1
    pitch_min: float = 50
    pitch_max: float = 400
    energy_min: float = 0
    energy_max: float = 1
    vad_min: float = 0
    vad_max: float = 1
    use_algorithmic_features: bool = False


@dataclass
class TIMITCollatorArgs:
    max_frames: int = 384
    vocex: str = "cdminix/vocex"
    vocex_fp16: bool = False
    name: str = "default_timit"
    mpm: str = None
    mpm_layer: int = -1
    pitch_min: float = 50
    pitch_max: float = 400
    energy_min: float = 0
    energy_max: float = 1
    vad_min: float = 0
    vad_max: float = 1
    use_algorithmic_features: bool = False


@dataclass
class BURNModelArgs:
    n_layers: int = 2
    hidden_dim: int = 512
    measures: str = "pitch,energy,voice_activity_binary"
    values_per_word: int = 10
    type: str = "mlp"  # can be "mlp" or "conformer"
    dropout: float = 0.1
    n_heads: int = 2
    kernel_size: int = 7
    max_length: int = 256


@dataclass
class RAVDESSModelArgs:
    n_layers: int = 2
    hidden_dim: int = 512
    measures: str = "pitch,energy,voice_activity_binary"
    type: str = "mlp"  # can be "mlp" or "transformer"
    dropout: float = 0.1
    n_heads: int = 2
    kernel_size: int = 7
    max_length: int = 512
    filter_size: int = 256


@dataclass
class TIMITModelArgs:
    n_layers: int = 2
    hidden_dim: int = 512
    measures: str = "pitch,energy,voice_activity_binary"
    type: str = "mlp"  # can be "mlp" or "transformer"
    dropout: float = 0.1
    n_heads: int = 2
    kernel_size: int = 7
    max_length: int = 384
    filter_size: int = 256


@dataclass
class HeatmapArgs:
    type: str = "intrinsic"  # can be "intrinsic" or "extrinsic"
    measure: str = "vad_mae"
    dataset: str = "burn"
