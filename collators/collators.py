from pathlib import Path

import torch
from torch import nn
import torchaudio
import numpy as np
import librosa
from vocex import Vocex

from configs.args import CollatorArgs

ALL_MEASURES = [
    "pitch",
    "energy",
    "voice_activity_binary",
]


def interpolate(x, target_length):
    x = np.interp(np.linspace(0, 1, target_length), np.linspace(0, 1, len(x)), x)
    return x


class BaselineBURNCollator:
    def __init__(
        self,
        args: CollatorArgs,
    ):
        """
        Collator for the baseline model, which extracts
        any subset of the following measures from the dataset:
        - pitch
        - energy
        - duration
        - voice activity
        """
        super().__init__()

        self.args = args
        self.vocex = Vocex.from_pretrained(self.args.vocex, fp16=self.args.vocex_fp16)

    def __call__(self, batch):
        results = {measure: [] for measure in self.args.measures.split(",")}
        # change batch from list of dicts to dict of lists
        batch = {k: [d[k] for d in batch] for k in batch[0]}
        for k, audio in enumerate(batch["audio"]):
            measures = {measure: [] for measure in ALL_MEASURES}
            for measure in ALL_MEASURES:
                file = Path(audio).with_suffix(f".{measure}.npy")
                if file.exists() and not self.args.overwrite:
                    measures[measure] = np.load(file)
            if (not file.exists()) or self.args.overwrite:
                file = Path(audio)
                audio, sr = librosa.load(audio, sr=16000)
                # create 6 second windows
                windows = []
                for i in range(0, len(audio), 96000):
                    windows.append(audio[i : i + 96000])
                for i, w in enumerate(windows):
                    vocex_output = self.vocex(w, sr)
                    for measure in ALL_MEASURES:
                        measures[measure].append(vocex_output["measures"][measure])
                for m, v in measures.items():
                    v = [v.flatten() for v in v]
                    v = np.concatenate(v)
                    v = torch.tensor(v).unsqueeze(0)
                    v = (
                        torchaudio.transforms.Resample(sr, 16000)(v)
                        .squeeze(0)
                        .T.numpy()
                    )
                    # min-max normalize
                    if (v.max() - v.min()) == 0:
                        v = np.zeros_like(v)
                    else:
                        v = (v - v.min()) / (v.max() - v.min())
                    measures[m] = v

                # take the mean according to batch["word_durations"] (an int in frames)
                durations = np.cumsum(batch["word_durations"][k])
                durations = np.insert(durations, 0, 0)
                # append last index to durations
                last_index = measures["pitch"].shape[0]
                durations = np.append(durations, last_index)
                for measure in ALL_MEASURES:
                    if self.args.values_per_word == 1:
                        measures[measure] = np.array(
                            [
                                np.mean(
                                    measures[measure][durations[i] : durations[i + 1]]
                                )
                                for i in range(len(durations) - 1)
                            ]
                        )
                    else:
                        measures[measure] = np.array(
                            [
                                interpolate(
                                    measures[measure][durations[i] : durations[i + 1]],
                                    self.args.values_per_word,
                                )
                                for i in range(len(durations) - 1)
                            ]
                        )
                    np.save(file.with_suffix(f".{measure}.npy"), measures[measure])
            for measure in results:
                results[measure].append(measures[measure])

        # pad to max length
        first_measure = list(results.keys())[0]
        if self.args.max_words is None:
            max_len = np.max([r.shape[0] for r in results[first_measure]])
        else:
            max_len = self.args.max_words

        mask = np.zeros((len(results[first_measure]), max_len))
        for i, r in enumerate(results[first_measure]):
            mask[i, : r.shape[0]] = 1

        batch["mask"] = torch.tensor(mask).bool()

        batch["measures"] = {measure: results[measure] for measure in results}

        # pad measures
        for measure in results:
            batch["measures"][measure] = torch.tensor(
                np.array(
                    [
                        np.pad(r, ((0, max_len - r.shape[0]), (0, 0)))
                        for r in batch["measures"][measure]
                    ]
                )
            ).to(torch.float32)
        # pad prominence and break
        batch["prominence"] = [np.array(p) for p in batch["prominence"]]
        batch["break"] = [np.array(b) for b in batch["break"]]
        batch["prominence"] = torch.tensor(
            np.array(
                [np.pad(p, (0, max_len - p.shape[0])) for p in batch["prominence"]]
            )
        )
        batch["break"] = torch.tensor(
            np.array([np.pad(b, (0, max_len - b.shape[0])) for b in batch["break"]])
        )
        # pad word durations
        batch["word_durations"] = [np.array(wd) for wd in batch["word_durations"]]
        batch["word_durations"] = torch.tensor(
            np.array(
                [
                    np.pad(wd, (0, max_len - wd.shape[0]))
                    for wd in batch["word_durations"]
                ]
            )
        )

        return batch
