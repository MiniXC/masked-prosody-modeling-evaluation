from .args import TrainingArgs, ModelArgs, BURNCollatorArgs, RAVDESSCollatorArgs


def validate_args(*args):
    for arg in args:
        if isinstance(arg, TrainingArgs):
            if arg.lr_schedule not in ["linear_with_warmup"]:
                raise ValueError(f"lr_schedule {arg.lr_schedule} not supported")
            if arg.wandb_mode not in ["online", "offline"]:
                raise ValueError(f"wandb_mode {arg.wandb_mode} not supported")
            if arg.wandb_mode == "online":
                if arg.wandb_project is None:
                    raise ValueError("wandb_project must be specified")
            if arg.push_to_hub:
                if arg.hub_repo is None:
                    raise ValueError("hub_repo must be specified")
        if isinstance(arg, ModelArgs):
            if arg.hidden_dim % 2 != 0:
                raise ValueError("hidden_dim should be divisible by 2")
        if isinstance(arg, BURNCollatorArgs) or isinstance(arg, RAVDESSCollatorArgs):
            if arg.name not in ["default_burn", "default_ravdess"]:
                raise ValueError(f"collator {arg.name} not supported")
            for measure in arg.measures.split(","):
                if measure not in [
                    "pitch",
                    "energy",
                    "duration",
                    "voice_activity_binary",
                ]:
                    raise ValueError(f"measure {measure} not supported")
