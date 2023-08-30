import sys

sys.path.append(".")

from wandb import Api
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from configs.args import HeatmapArgs

args = HfArgumentParser(HeatmapArgs).parse_args_into_dataclasses()[0]

api = Api()

if args.type == "intrinsic":
    runs = [
        r
        for r in api.runs("cdminix/masked-prosody-modeling")
        if ("bin" in r.name and "mask" in r.name and r.state == "finished")
    ]

    bins = []
    masks = []
    train_loss = []
    val_loss = []
    pitch_mae = []
    energy_mae = []
    vad_mae = []

    for r in tqdm(runs, desc="Loading runs"):
        bins.append(int(r.name.split("_")[0].replace("bin", "")))
        masks.append(int(r.name.split("_")[1].replace("mask", "")))
        train_loss.append(r.summary["train/loss"])
        val_loss.append(r.summary["val/loss"])
        pitch_mae.append(r.summary["val/mae_pitch"])
        energy_mae.append(r.summary["val/mae_energy"])
        vad_mae.append(r.summary["val/mae_vad"])

    df = pd.DataFrame(
        {
            "bins": bins,
            "masks": masks,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "pitch_mae": pitch_mae,
            "energy_mae": energy_mae,
            "vad_mae": vad_mae,
        }
    )

    df["val_minus_train"] = df["val_loss"] - df["train_loss"]

    value_to_plot = args.measure

    # Plot heatmap
    sns.heatmap(
        df.pivot(index="bins", columns="masks", values=value_to_plot),
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar=False,
    )

    plt.title(value_to_plot)
    plt.xlabel("Masks")
    plt.ylabel("Bins")
    plt.show()
elif args.type == "extrinsic":
    dataset = args.dataset
    runs = [
        r
        for r in api.runs("cdminix/masked-prosody-modeling")
        if (
            "bin" in r.name
            and "mask" in r.name
            and r.state == "finished"
            and dataset in r.name
            and "gs" in r.name
        )
    ]
    bins = []
    masks = []
    measure = []
    for r in tqdm(runs, desc="Loading runs"):
        bins.append(r.config["training"]["mpm_bin_size"])
        masks.append(r.config["training"]["mpm_mask_size"])
        # measure.append(r.summary[args.measure])
        measure.append(np.mean(r.history(keys=[args.measure])[args.measure].values))

    # min-max normalize measure
    measure = (measure - np.min(measure)) / (np.max(measure) - np.min(measure))

    df = pd.DataFrame(
        {
            "bins": bins,
            "masks": masks,
            "measure": measure,
        }
    )

    # Plot heatmap
    sns.heatmap(
        df.pivot(index="bins", columns="masks", values="measure"),
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar=False,
        vmin=0,
        vmax=1,
    )

    plt.title(f"{dataset} + {args.measure} + {args.type}")
    plt.xlabel("Masks")
    plt.ylabel("Bins")
    plt.show()
elif args.type == "full_f1":
    dfs = []
    for dataset, measure_name in [
        ("burn", "val/prom_f1"),
        ("burn", "val/break_f1"),
        ("ravdess", "val/f1"),
        ("timit", "val/phon_f1"),
        ("timit", "val/word_f1"),
    ]:
        runs = [
            r
            for r in api.runs("cdminix/masked-prosody-modeling")
            if (
                "bin" in r.name
                and "mask" in r.name
                and r.state == "finished"
                and dataset in r.name
                and "gs" in r.name
            )
        ]
        bins = []
        masks = []
        measure = []
        for r in tqdm(runs, desc="Loading runs"):
            bins.append(r.config["training"]["mpm_bin_size"])
            masks.append(r.config["training"]["mpm_mask_size"])
            # measure.append(r.summary[args.measure])
            measure.append(np.mean(r.history(keys=[measure_name])[measure_name].values))

        normalized_measure = (measure - np.min(measure)) / (
            np.max(measure) - np.min(measure)
        )
        df = pd.DataFrame(
            {
                "bins": bins,
                "masks": masks,
                "measure": normalized_measure,
            }
        )
        dfs.append(df)
    full_df = pd.concat(dfs)
    # Plot heatmap
    sns.heatmap(
        full_df.groupby(["bins", "masks"])
        .mean()
        .reset_index()
        .pivot(index="bins", columns="masks", values="measure"),
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar=False,
        vmin=0,
        vmax=1,
    )

    plt.title(f"Full + {args.type}")
    plt.xlabel("Masks")
    plt.ylabel("Bins")
    plt.show()
elif args.type == "full_loss":
    dfs = []
    for dataset, measure_name in [
        ("burn", "val/prom_loss"),
        ("burn", "val/break_loss"),
        ("ravdess", "val/loss"),
        ("timit", "val/phon_loss"),
        ("timit", "val/word_loss"),
    ]:
        runs = [
            r
            for r in api.runs("cdminix/masked-prosody-modeling")
            if (
                "bin" in r.name
                and "mask" in r.name
                and r.state == "finished"
                and dataset in r.name
                and "gs" in r.name
            )
        ]
        bins = []
        masks = []
        measure = []
        for r in tqdm(runs, desc="Loading runs"):
            bins.append(r.config["training"]["mpm_bin_size"])
            masks.append(r.config["training"]["mpm_mask_size"])
            # measure.append(r.summary[args.measure])
            measure.append(
                np.median(r.history(keys=[measure_name])[measure_name].values)
            )

        normalized_measure = (measure - np.min(measure)) / (
            np.max(measure) - np.min(measure)
        )
        df = pd.DataFrame(
            {
                "bins": bins,
                "masks": masks,
                "measure": normalized_measure,
            }
        )
        dfs.append(df)
    full_df = pd.concat(dfs)
    # Plot heatmap
    sns.heatmap(
        full_df.groupby(["bins", "masks"])
        .mean()
        .reset_index()
        .pivot(index="bins", columns="masks", values="measure"),
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar=False,
        vmin=0,
        vmax=1,
    )

    plt.title(f"Full + {args.type}")
    plt.xlabel("Masks")
    plt.ylabel("Bins")
    plt.show()
