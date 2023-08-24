import sys
import os

sys.path.append(".")

from pathlib import Path
from transformers import HfArgumentParser
from rich.console import Console
from datasets import load_dataset
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

console = Console()

from model.mpm.masked_prosody_model import MaskedProsodyModel
from configs.args import (
    PreprocessingArgs,
    BURNCollatorArgs,
    RAVDESSCollatorArgs,
    TIMITCollatorArgs,
)
from collators import get_collator
from scripts.util.plotting import plot_prosody_model_burn_batch


def main():
    parser = HfArgumentParser(PreprocessingArgs)
    args = parser.parse_args_into_dataclasses()[0]

    bins = args.bin_size
    mask = args.mask_size
    step = args.checkpoint_step

    console.rule("BURN")
    console.print("loading BURN dataset")
    train_ds = load_dataset(args.burn_dataset, split=args.burn_train_split)
    val_ds = load_dataset(args.burn_dataset, split=args.burn_val_split)

    console.print("loading BURN collator")
    burn_collator_args = BURNCollatorArgs()
    burn_collator_args.name = "prosody_model_burn"
    burn_collator_args.mpm = (
        f"masked-prosody-modeling/checkpoints/bin{bins}_mask{mask}/step_{step}"
    )
    burn_collator_args.overwrite = True
    burn_collator = get_collator(burn_collator_args)

    # console.print("preprocessing BURN dataset")
    # for ds in [train_ds, val_ds]:
    #     dl = DataLoader(
    #         ds, batch_size=1, collate_fn=burn_collator, num_workers=os.cpu_count()
    #     )
    #     for batch in tqdm(dl, desc="Preprocessing"):
    #         pass

    console.rule("RAVDESS")
    console.print("loading RAVDESS dataset")
    train_ds = load_dataset(args.ravdess_dataset, split=args.ravdess_train_split)
    val_ds = load_dataset(args.ravdess_dataset, split=args.ravdess_val_split)

    console.print("loading RAVDESS collator")
    ravdess_collator_args = RAVDESSCollatorArgs()
    ravdess_collator_args.name = "prosody_model_ravdess"
    ravdess_collator_args.mpm = (
        f"masked-prosody-modeling/checkpoints/bin{bins}_mask{mask}/step_{step}"
    )
    ravdess_collator_args.overwrite = True
    ravdess_collator = get_collator(ravdess_collator_args)

    console.print("preprocessing RAVDESS dataset")
    # for ds in [train_ds, val_ds]:
    #     dl = DataLoader(
    #         ds, batch_size=1, collate_fn=ravdess_collator, num_workers=os.cpu_count()
    #     )
    #     for batch in tqdm(dl, desc="Preprocessing"):
    #         pass

    console.rule("TIMIT")
    console.print("loading TIMIT dataset")
    train_ds = load_dataset(
        args.timit_dataset,
        split=args.timit_train_split,
        data_dir=os.environ["TIMIT_PATH"],
    )
    val_ds = load_dataset(
        args.timit_dataset,
        split=args.timit_val_split,
        data_dir=os.environ["TIMIT_PATH"],
    )

    console.print("loading TIMIT collator")
    timit_collator_args = TIMITCollatorArgs()
    timit_collator_args.name = "prosody_model_timit"
    timit_collator_args.mpm = (
        f"masked-prosody-modeling/checkpoints/bin{bins}_mask{mask}/step_{step}"
    )
    timit_collator_args.overwrite = True
    timit_collator = get_collator(timit_collator_args)

    console.print("preprocessing TIMIT dataset")
    for ds in [train_ds, val_ds]:
        dl = DataLoader(
            ds, batch_size=1, collate_fn=timit_collator, num_workers=os.cpu_count()
        )
        for batch in tqdm(dl, desc="Preprocessing"):
            pass


if __name__ == "__main__":
    main()
