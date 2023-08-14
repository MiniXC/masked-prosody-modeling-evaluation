import sys

sys.path.append(".")  # add root of project to path

from datasets import load_dataset
import numpy as np
import librosa

from collators.collators import BaselineRAVDESSCollator
from configs.args import CollatorArgs

dataset = load_dataset("narad/ravdess", split="train")

args = CollatorArgs()
args.measures = "pitch,energy,voice_activity_binary"
args.overwrite = True

collator = BaselineRAVDESSCollator(args)

batch = collator([dataset[0], dataset[1]])

import matplotlib.pyplot as plt
import seaborn as sns

sns.lineplot(
    x=range(len(batch["measures"]["pitch"][0])), y=batch["measures"]["pitch"][0] * 120
)
sns.lineplot(
    x=range(len(batch["measures"]["energy"][0])), y=batch["measures"]["energy"][0] * 120
)
sns.lineplot(
    x=range(len(batch["measures"]["voice_activity_binary"][0])),
    y=batch["measures"]["voice_activity_binary"][0] * 120,
)
# audio
audio = batch["audio"][0]["array"]
sr = batch["audio"][0]["sampling_rate"]
mels = np.log(librosa.feature.melspectrogram(y=audio, sr=sr) + 1e-6)
plt.imshow(mels, aspect="auto", origin="lower", interpolation="none")

plt.show()
