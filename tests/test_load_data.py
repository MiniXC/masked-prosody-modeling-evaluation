import os
import sys

sys.path.append(".")  # add root of project to path

from datasets import load_dataset
from torch.utils.data import DataLoader

from configs.args import (
    TrainingArgs,
    BURNCollatorArgs,
    RAVDESSCollatorArgs,
    TIMITCollatorArgs,
    BURNModelArgs,
    RAVDESSModelArgs,
    TIMITModelArgs,
)
from collators import get_collator

# BURN
default_args = TrainingArgs()
default_model_args_burn = BURNModelArgs()
default_collator_args_burn = BURNCollatorArgs()
default_collator_args_burn.measures = default_model_args_burn.measures
default_collator_args_burn.values_per_word = default_model_args_burn.values_per_word
default_collator_args_burn.overwrite = True

train_dataset_burn = load_dataset(
    default_args.burn_dataset, split=default_args.burn_train_split
)
val_dataset = load_dataset(default_args.burn_dataset, split=default_args.burn_val_split)

collator = get_collator(default_collator_args_burn)

dataloader_burn = DataLoader(
    train_dataset_burn,
    batch_size=default_args.batch_size,
    shuffle=True,
    collate_fn=collator,
)


def test_dataloader_burn():
    if os.environ.get("BURN_PATH") is not None:
        for batch in dataloader_burn:
            assert batch["measures"]["pitch"].shape == (
                default_args.batch_size,
                default_collator_args_burn.max_words,
                default_collator_args_burn.values_per_word,
            )
            break
