from dataclasses import dataclass


@dataclass
class TrainingArgs:
    lr: float = 1e-4
    lr_schedule: str = "linear_with_warmup"
    lr_warmup_steps: int = 1000
    gradient_clip_val: float = 1.0
    checkpoint_path: str = "checkpoints"
    output_path: str = "outputs"
    run_name: str = None
    wandb_mode: str = "offline"
    wandb_project: str = None
    wandb_dir: str = "wandb"
    train_split: str = "train[:90%]"
    val_split: str = "train[90%:]"
    n_steps: int = 2000
    batch_size: int = 32
    seed: int = 0
    burn_dataset: str = "cdminix/bu_radio"
    ravdess_dataset: str = "narad/ravdess"
    log_every_n_steps: int = 10
    do_full_eval: bool = True
    do_save: bool = False
    save_onnx: bool = False
    eval_only: bool = False
    eval_every_n_steps: int = 100
    save_every_n_steps: int = 1000
    push_to_hub: bool = False
    hub_repo: str = None


@dataclass
class BURNCollatorArgs:
    overwrite: bool = False
    max_words: int = 256
    name: str = "default_burn"
    vocex: str = "cdminix/vocex"
    vocex_fp16: bool = False


@dataclass
class RAVDESSCollatorArgs:
    overwrite: bool = False
    max_frames: int = 512
    vocex: str = "cdminix/vocex"
    vocex_fp16: bool = False
    name: str = "default_ravdess"


@dataclass
class ModelArgs:
    n_layers: int = 4
    hidden_dim: int = 512
    measures: str = "pitch,energy,voice_activity_binary"
    values_per_word: int = 10
    type: str = "mlp"  # can be "mlp" or "transformer"
    dropout: float = 0.1
