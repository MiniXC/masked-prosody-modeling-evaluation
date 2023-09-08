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
        and "gs" not in r.name
        and "diff" not in r.name
        and "var" not in r.name
    ]

    bins = []
    masks = []
    train_loss = []
    val_loss = []
    pitch_mae = []
    energy_mae = []
    vad_mae = []

    for r in tqdm(runs, desc="Loading runs"):
        print(r.name)
        bins.append(int(r.name.split("_")[0].replace("bin", "")))
        masks.append(int(r.name.split("_")[1].replace("mask", "")))
        train_loss_vals = r.history(keys=["train/loss"])["train/loss"].values
        first_time_within_20_percent = np.where(
            train_loss_vals < train_loss_vals[-1] * 1.2
        )[0][0]
        train_loss.append(first_time_within_20_percent / len(train_loss_vals))
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
    plt.xlabel("Mask Size")
    plt.ylabel("Codebook Size")
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
            and "var" not in r.name
            and "diff" not in r.name
        )
    ]
    bins = []
    masks = []
    measure = []
    for r in tqdm(runs, desc="Loading runs"):
        bins.append(r.config["training"]["mpm_bin_size"])
        masks.append(r.config["training"]["mpm_mask_size"])
        # measure.append(r.summary[args.measure])
        measure.append(np.max(r.history(keys=[args.measure])[args.measure].values))

    # measure = (measure - np.mean(measure)) / np.std(measure) * -1

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
        cmap="RdYlGn",
        cbar=False,
    )

    plt.xlabel("Mask Size")
    plt.ylabel("Codebook Size")
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
                and "var" not in r.name
                and "diff" not in r.name
            )
        ]
        bins = []
        masks = []
        measure = []
        for r in tqdm(runs, desc="Loading runs"):
            bins.append(r.config["training"]["mpm_bin_size"])
            masks.append(r.config["training"]["mpm_mask_size"])
            # measure.append(r.summary[args.measure])
            measure.append(np.max(r.history(keys=[measure_name])[measure_name].values))

        normalized_measure = (measure - np.mean(measure)) / np.std(measure)
        df = pd.DataFrame(
            {
                "bins": bins,
                "masks": masks,
                "measure": normalized_measure,
                "unnormalized_measure": measure,
            }
        )
        print(
            measure_name,
            df["unnormalized_measure"].mean(),
            df["unnormalized_measure"].std(),
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
        fmt=".3f",  # divergent color map (red = bad, green = good) with white at 0
        cmap="RdYlGn",
        cbar=False,
        vmin=-1,
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
                and "var" not in r.name
                and "diff" not in r.name
            )
        ]
        bins = []
        masks = []
        measure = []
        for r in tqdm(runs, desc="Loading runs"):
            bins.append(r.config["training"]["mpm_bin_size"])
            masks.append(r.config["training"]["mpm_mask_size"])
            # measure.append(r.summary[args.measure])
            measure.append(np.min(r.history(keys=[measure_name])[measure_name].values))

        normalized_measure = (measure - np.mean(measure)) / np.std(measure)
        df = pd.DataFrame(
            {
                "bins": bins,
                "masks": masks,
                "measure": normalized_measure,
                "unnormalized_measure": measure,
            }
        )
        dfs.append(df)

        print(
            measure_name,
            df["unnormalized_measure"].mean(),
            df["unnormalized_measure"].std(),
        )
    full_df = pd.concat(dfs)
    # Plot heatmap
    sns.heatmap(
        full_df.groupby(["bins", "masks"])
        .mean()
        .reset_index()
        .pivot(index="bins", columns="masks", values="measure"),
        annot=True,
        fmt=".3f",
        cmap="RdYlGn_r",
        cbar=False,
        vmin=-1,
        vmax=1,
    )

    plt.xlabel("Mask Size")
    plt.ylabel("Codebook Size")
    plt.show()
elif args.type == "layers":
    runs = [
        r
        for r in api.runs("cdminix/masked-prosody-modeling")
        if ("_layer" in r.name and args.dataset in r.name)
    ]
    layers = []
    measure = []
    for r in tqdm(runs, desc="Loading runs"):
        print(r.name)
        layers.append(int(r.name.split("_")[-1]))
        measure.append(r.summary[args.measure])

    df = pd.DataFrame(
        {
            "layers": layers,
            "measure": measure,
        }
    )

    # Plot line
    sns.lineplot(data=df, x="layers", y="measure")
    plt.show()
elif args.type == "layers_all":
    dfs = []
    task_dict = {
        "val/prom_f1": "Prominence",
        "val/break_f1": "Boundary",
    }
    for dataset, measure_name in [
        ("burn", "val/prom_f1"),
        ("burn", "val/break_f1"),
    ]:
        runs = [
            r
            for r in api.runs("cdminix/masked-prosody-modeling")
            if ("_layer" in r.name and dataset in r.name)
        ]
        layers = []
        measure = []
        for r in tqdm(runs, desc="Loading runs"):
            layers.append(int(r.name.split("_")[-1]) + 1)
            measure.append(
                np.median(r.history(keys=[measure_name])[measure_name].values)
            )

        df = pd.DataFrame(
            {
                "layers": layers,
                "measure": measure,
            }
        )
        df["Task"] = task_dict[measure_name]
        dfs.append(df)

    # subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.5))
    sns.lineplot(
        data=dfs[0],
        x="layers",
        y="measure",
        hue="Task",
        ax=axes[0],
        marker="o",
        palette=("#6C8EBF",),
    )
    sns.lineplot(
        data=dfs[1],
        x="layers",
        y="measure",
        hue="Task",
        ax=axes[1],
        marker="o",
        palette=("#D6B656",),
    )
    axes[0].set_title("Burn (Prominence)")
    axes[0].set_xticks(dfs[0]["layers"])
    axes[0].legend(loc="lower right")
    axes[0].set_ylabel("F1")
    axes[0].set_xlabel("Layer")
    axes[1].set_title("Burn (Boundary)")
    axes[1].set_xticks(dfs[1]["layers"])
    axes[1].legend(loc="lower right")
    axes[1].set_ylabel("F1")
    axes[1].set_xlabel("Layer")
    sns.despine()
    plt.show()
