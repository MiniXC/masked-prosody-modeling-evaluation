from wandb import Api
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

api = Api()

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

value_to_plot = "vad_mae"

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
