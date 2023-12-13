from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import librosa
from vocex import Vocex

from model.mpm.masked_prosody_model import MaskedProsodyModel

# from model.mpm.masked_prosody_model import (
#     ConversationalMaskedProsodyModel as MaskedProsodyModel,
# )
from speech_collator.measures import (
    PitchMeasure,
    EnergyMeasure,
    VoiceActivityMeasure,
)
from scipy.signal import cwt, ricker

from configs.args import BURNCollatorArgs, RAVDESSCollatorArgs, SWBCollatorArgs, TIMITCollatorArgs

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
        args: BURNCollatorArgs,
        device: torch.device = None,
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
        if args.use_algorithmic_features:
            self.pitch_measure = PitchMeasure()
            self.energy_measure = EnergyMeasure()
            self.voice_activity_measure = VoiceActivityMeasure()

    def __call__(self, batch):
        results = {measure: [] for measure in self.args.measures.split(",")}
        # change batch from list of dicts to dict of lists
        batch = {k: [d[k] for d in batch] for k in batch[0]}
        for k, audio in enumerate(batch["audio"]):
            measures = {measure: [] for measure in ALL_MEASURES}
            audio_path = audio
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
                    if self.args.use_algorithmic_features:
                        # resample w to 22050
                        w = librosa.resample(w, orig_sr=sr, target_sr=22050)
                        pitch = self.pitch_measure(w, np.array([1000]))["measure"]
                        energy = self.energy_measure(w, np.array([1000]))["measure"]
                        vad = self.voice_activity_measure(w, np.array([1000]))[
                            "measure"
                        ]
                        if self.args.use_cwt:
                            pitch = (pitch - pitch.mean()) / (pitch.std() + 1e-8)
                            energy = (energy - energy.mean()) / (energy.std() + 1e-8)
                            vad = (vad - vad.mean()) / (vad.std() + 1e-8)
                            pitch = cwt(
                                pitch, ricker, np.arange(1, self.args.cwt_n_bins + 1)
                            ).T
                            energy = cwt(
                                energy, ricker, np.arange(1, self.args.cwt_n_bins + 1)
                            ).T
                            vad = cwt(
                                vad, ricker, np.arange(1, self.args.cwt_n_bins + 1)
                            ).T
                        vocex_output = {
                            "measures": {
                                "pitch": torch.tensor(pitch),
                                "energy": torch.tensor(energy),
                                "voice_activity_binary": torch.tensor(vad),
                            }
                        }
                    else:
                        vocex_output = self.vocex(w, sr)
                    for measure in ALL_MEASURES:
                        measures[measure].append(vocex_output["measures"][measure])
                for m, v in measures.items():
                    if not self.args.use_cwt:
                        v = [v.flatten() for v in v]
                        v = np.concatenate(v)
                        v = torch.tensor(v).unsqueeze(0)
                        v = (
                            torchaudio.transforms.Resample(sr, 16000)(v)
                            .squeeze(0)
                            .T.numpy()
                        )
                        # normalize
                        if (v.max() - v.min()) == 0:
                            v = np.zeros_like(v)
                        else:
                            if m == "pitch":
                                v = np.clip(
                                    v, self.args.pitch_min, self.args.pitch_max
                                ) / (self.args.pitch_max - self.args.pitch_min)
                            elif m == "energy":
                                v = np.clip(
                                    v, self.args.energy_min, self.args.energy_max
                                ) / (self.args.energy_max - self.args.energy_min)
                            elif m == "voice_activity_binary":
                                v = np.clip(v, self.args.vad_min, self.args.vad_max) / (
                                    self.args.vad_max - self.args.vad_min
                                )
                    else:
                        v = [v.numpy() for v in v]
                        v = np.concatenate(v)
                        v = torch.from_numpy(v)
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
                                    measures[measure][durations[i] : durations[i + 1]],
                                    axis=0,
                                )
                                for i in range(len(durations) - 1)
                            ]
                        )
                    else:
                        if not self.args.use_cwt:
                            measures[measure] = np.array(
                                [
                                    interpolate(
                                        measures[measure][
                                            durations[i] : durations[i + 1]
                                        ],
                                        self.args.values_per_word,
                                    )
                                    for i in range(len(durations) - 1)
                                ]
                            )
                        else:
                            dims = measures[measure].shape[-1]
                            # interpolate for each dimension
                            measures[measure] = np.array(
                                [
                                    np.stack(
                                        [
                                            interpolate(
                                                measures[measure][
                                                    durations[i] : durations[i + 1],
                                                    j,
                                                ],
                                                self.args.values_per_word,
                                            )
                                            for j in range(dims)
                                        ]
                                    )
                                    for i in range(len(durations) - 1)
                                ]
                            )
                    # collapse last two dimensions
                    measures[measure] = measures[measure].reshape(
                        measures[measure].shape[0], -1
                    )
                    np.save(
                        Path(audio_path).with_suffix(f".{measure}.npy"),
                        measures[measure],
                    )
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


class ProsodyModelBURNCollator:
    def __init__(
        self,
        args: BURNCollatorArgs,
        device: torch.device = None,
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
        self.suffix = "mpm"
        self.vocex = Vocex.from_pretrained(self.args.vocex, fp16=self.args.vocex_fp16)
        self.mpm = MaskedProsodyModel.from_pretrained(self.args.mpm)
        if self.args.use_mpm_random: # Reset the pretrained model weights to random init
            print(f"E.g. weights before init:\n{self.mpm.state_dict()['transformer.layers.0.self_attn.out_proj.weight']}")
            self.mpm.apply(self.mpm._init_all_weights)
            self.suffix = "mpm_rand"
            print(f"Same weights after init:\n{self.mpm.state_dict()['transformer.layers.0.self_attn.out_proj.weight']}")
        self.mpm.args.max_length = 512
        if device is not None:
            self.mpm.to(device)
        self.device = device
        self.bins = torch.linspace(0, 1, self.mpm.args.bins)
        if args.use_algorithmic_features:
            self.pitch_measure = PitchMeasure()
            self.energy_measure = EnergyMeasure()
            self.voice_activity_measure = VoiceActivityMeasure()

    def __call__(self, batch):
        results = []

        batch = {k: [d[k] for d in batch] for k in batch[0]}
        for k, audio in enumerate(batch["audio"]):
            mpms = []
            audio_path = audio
            file = Path(audio).with_suffix(f".{self.suffix}.npy")
            if (not file.exists()) or self.args.overwrite:
                file = Path(audio)
                audio, sr = librosa.load(audio, sr=16000)
                # create 6 second windows
                windows = []
                for i in range(0, len(audio), 96000):
                    windows.append(audio[i : i + 96000])
                for i, w in enumerate(windows):
                    if not self.args.use_algorithmic_features:
                        vocex_output = self.vocex(w, sr)
                        pitch = vocex_output["measures"]["pitch"]
                        energy = vocex_output["measures"]["energy"]
                        vad = vocex_output["measures"]["voice_activity_binary"]
                    else:
                        # resample w to 22050
                        w = librosa.resample(w, orig_sr=sr, target_sr=22050)
                        pitch = self.pitch_measure(w, np.array([1000]))["measure"]
                        energy = self.energy_measure(w, np.array([1000]))["measure"]
                        vad = self.voice_activity_measure(w, np.array([1000]))[
                            "measure"
                        ]
                    # normalize using pitch_min, pitch_max, energy_min, energy_max, vad_min, vad_max
                    if not isinstance(pitch, torch.Tensor):
                        pitch = torch.tensor(pitch)
                    if not isinstance(energy, torch.Tensor):
                        energy = torch.tensor(energy)
                    if not isinstance(vad, torch.Tensor):
                        vad = torch.tensor(vad)
                    pitch = torch.clip(
                        pitch.unsqueeze(0),
                        self.args.pitch_min,
                        self.args.pitch_max,
                    ) / (self.args.pitch_max - self.args.pitch_min)
                    energy = torch.clip(
                        energy.unsqueeze(0),
                        self.args.energy_min,
                        self.args.energy_max,
                    ) / (self.args.energy_max - self.args.energy_min)
                    vad = torch.clip(
                        vad.unsqueeze(0),
                        self.args.vad_min,
                        self.args.vad_max,
                    )
                    # bucketize
                    pitch = torch.bucketize(pitch, self.bins) + 2
                    pitch[pitch == self.mpm.args.bins + 2] = self.mpm.args.bins + 1
                    energy = torch.bucketize(energy, self.bins) + 2
                    vad = torch.bucketize(vad, torch.linspace(0, 1, 2)) + 2
                    mpm_input = torch.stack(
                        [
                            pitch,
                            energy,
                            vad,
                        ]
                    ).transpose(0, 1)
                    if self.device is not None:
                        mpm_input = mpm_input.to(self.device)
                    mpm_input = mpm_input.squeeze(2)
                    mpm_input = mpm_input[:, :, : self.mpm.args.max_length]
                    prev_len = mpm_input.shape[-1]
                    if mpm_input.shape[-1] < self.mpm.args.max_length:
                        zs = torch.zeros(
                            (
                                mpm_input.shape[0],
                                mpm_input.shape[1],
                                self.mpm.args.max_length - prev_len,
                            )
                        )
                        if self.device is not None:
                            zs = zs.to(self.device)
                        mpm_input = torch.cat(
                            [
                                mpm_input,
                                zs,
                            ],
                            dim=-1,
                        )
                    # note: self.long() is equivalent to self.to(torch.int64).
                    mpms.append(mpm_input.long())
                mpms_batch = torch.stack(mpms, dim=1).squeeze(0)

                mpm_output = self.mpm(mpms_batch, return_layer=self.args.mpm_layer)
                mpms = mpm_output["representations"].detach().cpu()
                # concat to go from (batch_size, len, hidden_dim) to (len, hidden_dim)
                mpms = mpms.reshape(-1, mpms.shape[-1])
                word_mpms = []
                # take the mean according to batch["word_durations"] (an int in frames)
                durations = np.cumsum(batch["word_durations"][k])
                durations = np.insert(durations, 0, 0)
                # append last index to durations
                last_index = mpms.shape[0]
                durations = np.append(durations, last_index)
                for i in range(len(durations) - 1):
                    # mean-max pool over the word
                    word_mpms.append(
                        torch.concat(
                            [
                                mpms[durations[i] : durations[i + 1]].mean(0),
                                mpms[durations[i] : durations[i + 1]].max(0).values,
                            ]
                        )
                    )
                word_mpms = torch.stack(word_mpms)
                np.save(
                    Path(audio_path).with_suffix(f".mpm.npy"),
                    word_mpms,
                )
            else:
                word_mpms = np.load(file)
            results.append(word_mpms) 

        # pad to max length
        if self.args.max_words is None:
            max_len = np.max([r.shape[0] for r in results])
        else:
            max_len = self.args.max_words

        mask = np.zeros((len(results), max_len))
        for i, r in enumerate(results):
            mask[i, : r.shape[0]] = 1

        batch["mask"] = torch.tensor(mask).bool()

        batch["mpm"] = torch.tensor(
            np.array([np.pad(r, ((0, max_len - r.shape[0]), (0, 0))) for r in results])
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


class ProsodyModelSWBCollator:
    """Copy from BURN collator but uses frame indices directly from dataset rather than computing them explicitly"""
    def __init__(
        self,
        args: SWBCollatorArgs,
        device: torch.device = None,
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
        self.suffix = 'mpm'
        self.vocex = Vocex.from_pretrained(self.args.vocex, fp16=self.args.vocex_fp16)
        self.mpm = MaskedProsodyModel.from_pretrained(self.args.mpm)
        if self.args.use_mpm_random: # Reset the pretrained model weights to random init
            print(f"E.g. weights before init:\n{self.mpm.state_dict()['transformer.layers.0.self_attn.out_proj.weight']}")
            self.mpm.apply(self.mpm._init_all_weights)
            self.suffix = 'mpm_rand'
            print(f"Same weights after init:\n{self.mpm.state_dict()['transformer.layers.0.self_attn.out_proj.weight']}")
        if device is not None:
            self.mpm.to(device)
        self.device = device
        self.bins = torch.linspace(0, 1, self.mpm.args.bins)
        if args.use_algorithmic_features:
            self.pitch_measure = PitchMeasure()
            self.energy_measure = EnergyMeasure()
            self.voice_activity_measure = VoiceActivityMeasure()

    def __call__(self, batch):
        results = []
        batch = {k: [d[k] for d in batch] for k in batch[0]}
        for k, audio in enumerate(batch["audio"]): #tqdm(enumerate(batch["audio"]), desc="Processing batch"):
            mpms = []
            audio_path = audio
            file = Path(audio).with_suffix(f".{self.suffix}.npy")
            if (not file.exists()) or self.args.overwrite:
                file = Path(audio)
                audio, sr = librosa.load(audio, sr=16000)
                # create 6 second windows
                windows = []
                for i in range(0, len(audio), 96000):
                    windows.append(audio[i : i + 96000])
                # Get audio features of entire audio (computed per window)
                for i, w in enumerate(windows):
                    if not self.args.use_algorithmic_features:
                        vocex_output = self.vocex(w, sr)
                        pitch = vocex_output["measures"]["pitch"]
                        energy = vocex_output["measures"]["energy"]
                        vad = vocex_output["measures"]["voice_activity_binary"]
                    else:
                        # resample w to 22050 # TODO ??? 
                        w = librosa.resample(w, orig_sr=sr, target_sr=22050)
                        pitch = self.pitch_measure(w, np.array([1000]))["measure"]
                        energy = self.energy_measure(w, np.array([1000]))["measure"]
                        vad = self.voice_activity_measure(w, np.array([1000]))[
                            "measure"
                        ]
                    # normalize using pitch_min, pitch_max, energy_min, energy_max, vad_min, vad_max
                    pitch = torch.clip(
                        torch.tensor(pitch).unsqueeze(0),
                        self.args.pitch_min,
                        self.args.pitch_max,
                    ) / (self.args.pitch_max - self.args.pitch_min)
                    energy = torch.clip(
                        torch.tensor(energy).unsqueeze(0),
                        self.args.energy_min,
                        self.args.energy_max,
                    ) / (self.args.energy_max - self.args.energy_min)
                    vad = torch.clip(
                        torch.tensor(vad).unsqueeze(0),
                        self.args.vad_min,
                        self.args.vad_max,
                    )
                    # # bucketize
                    # torch.bucketize(pitch, self.bins) + 2
                    # torch.bucketize(energy, self.bins) + 2
                    # torch.bucketize(vad, torch.linspace(0, 1, 2)) + 2
                    # bucketize
                    pitch = torch.bucketize(pitch, self.bins) + 2
                    pitch[pitch == self.mpm.args.bins + 2] = self.mpm.args.bins + 1
                    energy = torch.bucketize(energy, self.bins) + 2
                    vad = torch.bucketize(vad, torch.linspace(0, 1, 2)) + 2
                    mpm_input = torch.stack(
                        [
                            pitch,
                            energy,
                            vad,
                        ]
                    ).transpose(0, 1)
                    if self.device is not None:
                        mpm_input = mpm_input.to(self.device)

                    # Slice to max_length, else pad
                    mpm_input = mpm_input.squeeze(2)
                    mpm_input = mpm_input[:, :, : self.mpm.args.max_length]
                    prev_len = mpm_input.shape[-1]
                    if mpm_input.shape[-1] < self.mpm.args.max_length:
                        zs = torch.zeros(
                            (
                                mpm_input.shape[0],
                                mpm_input.shape[1],
                                self.mpm.args.max_length - prev_len,
                            )
                        )
                        if self.device is not None:
                            zs = zs.to(self.device)
                        mpm_input = torch.cat(
                            [
                                mpm_input,
                                zs,
                            ],
                            dim=-1,
                        )
                    # NB: self.long() is equivalent to self.to(torch.int64).
                    mpms.append(mpm_input.long())
                mpms_batch = torch.stack(mpms, dim=1).squeeze(0)

                # Get MPM features for each window over entire audio sequence
                mpm_output = self.mpm(mpms_batch, return_layer=self.args.mpm_layer)
                mpms = mpm_output["representations"].detach().cpu()
                # concat to go from (batch_size, len, hidden_dim) to (len, hidden_dim)
                mpms = mpms.reshape(-1, mpms.shape[-1])
                
                # Get word-level representations directly from frame indices
                word_mpms = []
                durations = batch["word_frames"][k]
                for i in range(len(durations) - 1):
                    # mean-max pool over the word
                    word_mpms.append(
                        torch.concat(
                            [
                                mpms[durations[i][0] : durations[i][1]].mean(0),
                                mpms[durations[i][0] : durations[i][1]].max(0).values,
                            ]
                        )
                    )
                word_mpms = torch.stack(word_mpms)
                np.save(
                    Path(audio_path).with_suffix(f".mpm.npy"),
                    word_mpms,
                )
            else:
                word_mpms = np.load(file)
            results.append(word_mpms)

        # pad to max length
        if self.args.max_words is None:
            max_len = np.max([r.shape[0] for r in results])
        else:
            max_len = self.args.max_words

        mask = np.zeros((len(results), max_len))
        for i, r in enumerate(results):
            mask[i, : r.shape[0]] = 1

        try:
            batch["mask"] = torch.tensor(mask).bool()
            batch["mpm"] = torch.tensor(
                np.array([np.pad(r, ((0, max_len - r.shape[0]), (0, 0))) for r in results])
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
            batch["word_frames"] = [np.array(wd) for wd in batch["word_frames"]]
            # Removing this from the batch as 2D array ruins padding and it isn't used during training 
            # batch["word_frames"] = torch.tensor(
            #     np.array(
            #         [
            #             np.pad(wd, ([0,0], max_len - wd.shape[0])) for wd in batch["word_frames"]
            #         ]
            #     )
            # )
        except Exception as error:
            print(error)
            import IPython
            IPython.embed()
        return batch


class BaselineRAVDESSCollator:
    def __init__(
        self,
        args: RAVDESSCollatorArgs,
        device: torch.device = None,
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
        self.emotions2int = {
            "neutral": 0,
            "calm": 1,
            "happy": 2,
            "sad": 3,
            "angry": 4,
            "fearful": 5,
            "disgust": 6,
            "surprised": 7,
        }
        if args.use_algorithmic_features:
            self.pitch_measure = PitchMeasure()
            self.energy_measure = EnergyMeasure()
            self.voice_activity_measure = VoiceActivityMeasure()

    def __call__(self, batch):
        results = {measure: [] for measure in self.args.measures.split(",")}
        # change batch from list of dicts to dict of lists
        batch = {k: [d[k] for d in batch] for k in batch[0]}
        for k, audio in enumerate(batch["audio"]):
            audio_path = audio["path"]
            measures = {measure: [] for measure in ALL_MEASURES}
            for measure in ALL_MEASURES:
                file = Path(audio_path).with_suffix(f".{measure}.npy")
                if file.exists() and not self.args.overwrite:
                    measures[measure] = np.load(file)
            if (not file.exists()) or self.args.overwrite:
                audio, sr = audio["array"], audio["sampling_rate"]
                if self.args.use_algorithmic_features:
                    # resample w to 22050
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
                    pitch = self.pitch_measure(audio, np.array([1000]))["measure"]
                    energy = self.energy_measure(audio, np.array([1000]))["measure"]
                    vad = self.voice_activity_measure(audio, np.array([1000]))[
                        "measure"
                    ]
                    if self.args.use_cwt:
                        pitch = (pitch - pitch.mean()) / (pitch.std() + 1e-8)
                        energy = (energy - energy.mean()) / (energy.std() + 1e-8)
                        vad = (vad - vad.mean()) / (vad.std() + 1e-8)
                        pitch = cwt(
                            pitch, ricker, np.arange(1, self.args.cwt_n_bins + 1)
                        ).T
                        energy = cwt(
                            energy, ricker, np.arange(1, self.args.cwt_n_bins + 1)
                        ).T
                        vad = cwt(vad, ricker, np.arange(1, self.args.cwt_n_bins + 1)).T
                    vocex_output = {
                        "measures": {
                            "pitch": torch.tensor(pitch),
                            "energy": torch.tensor(energy),
                            "voice_activity_binary": torch.tensor(vad),
                        }
                    }
                else:
                    vocex_output = self.vocex(audio, sr)
                for measure in ALL_MEASURES:
                    if not self.args.use_cwt:
                        v = vocex_output["measures"][measure]
                        v = v.flatten()
                        # min-max normalize
                        if (v.max() - v.min()) == 0:
                            v = np.zeros_like(v)
                        else:
                            if measure == "pitch":
                                v = np.clip(
                                    v, self.args.pitch_min, self.args.pitch_max
                                ) / (self.args.pitch_max - self.args.pitch_min)
                            elif measure == "energy":
                                v = np.clip(
                                    v, self.args.energy_min, self.args.energy_max
                                ) / (self.args.energy_max - self.args.energy_min)
                            elif measure == "voice_activity_binary":
                                v = np.clip(v, self.args.vad_min, self.args.vad_max) / (
                                    self.args.vad_max - self.args.vad_min
                                )
                    else:
                        v = vocex_output["measures"][measure]
                        v = v.numpy()

                    measures[measure] = v

                    np.save(
                        Path(audio_path).with_suffix(f".{measure}.npy"),
                        v,
                    )
            for measure in results:
                results[measure].append(measures[measure])

        # pad to max length
        first_measure = list(results.keys())[0]
        if self.args.max_frames is None:
            max_len = np.max([r.shape[0] for r in results[first_measure]])
        else:
            max_len = self.args.max_frames

        mask = np.zeros((len(results[first_measure]), max_len))

        for i, r in enumerate(results[first_measure]):
            mask[i, : r.shape[0]] = 1

        batch["mask"] = torch.tensor(mask).bool()

        batch["measures"] = {measure: results[measure] for measure in results}

        # pad measures
        for measure in results:
            if not self.args.use_cwt:
                batch["measures"][measure] = torch.tensor(
                    np.array(
                        [
                            np.pad(r, (0, max_len - r.shape[0]))
                            for r in batch["measures"][measure]
                        ]
                    )
                ).to(torch.float32)
            else:
                dims = batch["measures"][measure][0].shape[-1]
                # interpolate for each dimension
                batch["measures"][measure] = (
                    torch.tensor(
                        np.array(
                            [
                                np.stack(
                                    [
                                        interpolate(
                                            r[:, j],
                                            max_len,
                                        )
                                        for j in range(dims)
                                    ]
                                )
                                for r in batch["measures"][measure]
                            ]
                        )
                    )
                    .to(torch.float32)
                    .transpose(1, 2)
                )

        batch["emotion"] = torch.tensor(batch["labels"]).long()
        batch["emotion_onehot"] = F.one_hot(
            batch["emotion"], num_classes=len(self.emotions2int)
        ).to(torch.float32)

        return batch


class ProsodyModelRAVDESSCollator:
    def __init__(
        self,
        args: RAVDESSCollatorArgs,
        device: torch.device = None,
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
        self.emotions2int = {
            "neutral": 0,
            "calm": 1,
            "happy": 2,
            "sad": 3,
            "angry": 4,
            "fearful": 5,
            "disgust": 6,
            "surprised": 7,
        }
        self.mpm = MaskedProsodyModel.from_pretrained(self.args.mpm)
        if self.args.use_mpm_random: # Reset the pretrained model weights to random init
            print(f"E.g. weights before init:\n{self.mpm.state_dict()['transformer.layers.0.self_attn.out_proj.weight']}")
            self.mpm.apply(self.mpm._init_all_weights)
            print(f"Same weights after init:\n{self.mpm.state_dict()['transformer.layers.0.self_attn.out_proj.weight']}")
        if device is not None:
            self.mpm.to(device)
        self.device = device
        self.bins = torch.linspace(0, 1, self.mpm.args.bins)
        if args.use_algorithmic_features:
            self.pitch_measure = PitchMeasure()
            self.energy_measure = EnergyMeasure()
            self.voice_activity_measure = VoiceActivityMeasure()

    def __call__(self, batch):
        results = []
        # change batch from list of dicts to dict of lists
        batch = {k: [d[k] for d in batch] for k in batch[0]}
        for k, audio in enumerate(batch["audio"]):
            audio_path = audio["path"]
            file = Path(audio_path).with_suffix(f".mpm.npy")
            if (not file.exists()) or self.args.overwrite:
                audio, sr = audio["array"], audio["sampling_rate"]
                if not self.args.use_algorithmic_features:
                    vocex_output = self.vocex(audio, sr)
                    pitch = vocex_output["measures"]["pitch"]
                    energy = vocex_output["measures"]["energy"]
                    vad = vocex_output["measures"]["voice_activity_binary"]
                else:
                    # resample w to 22050
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
                    pitch = self.pitch_measure(audio, np.array([1000]))["measure"]
                    energy = self.energy_measure(audio, np.array([1000]))["measure"]
                    vad = self.voice_activity_measure(audio, np.array([1000]))[
                        "measure"
                    ]
                # normalize using pitch_min, pitch_max, energy_min, energy_max, vad_min, vad_max
                if not isinstance(pitch, torch.Tensor):
                    pitch = torch.tensor(pitch)
                if not isinstance(energy, torch.Tensor):
                    energy = torch.tensor(energy)
                if not isinstance(vad, torch.Tensor):
                    vad = torch.tensor(vad)
                pitch = torch.clip(
                    pitch.unsqueeze(0),
                    self.args.pitch_min,
                    self.args.pitch_max,
                ) / (self.args.pitch_max - self.args.pitch_min)
                energy = torch.clip(
                    energy.unsqueeze(0),
                    self.args.energy_min,
                    self.args.energy_max,
                ) / (self.args.energy_max - self.args.energy_min)
                vad = torch.clip(vad.unsqueeze(0), self.args.vad_min, self.args.vad_max)
                # bucketize
                pitch = torch.bucketize(pitch, self.bins) + 2
                pitch[pitch == self.mpm.args.bins + 2] = self.mpm.args.bins + 1
                energy = torch.bucketize(energy, self.bins) + 2
                vad = torch.bucketize(vad, torch.linspace(0, 1, 2)) + 2
                mpm_input = torch.stack(
                    [
                        pitch,
                        energy,
                        vad,
                    ]
                ).transpose(0, 1)
                if self.device is not None:
                    mpm_input = mpm_input.to(self.device)
                mpm_input = mpm_input.squeeze(2)
                prev_len = mpm_input.shape[-1]
                if mpm_input.shape[-1] < self.mpm.args.max_length:
                    mpm_input = torch.cat(
                        [
                            mpm_input,
                            torch.zeros(
                                (
                                    mpm_input.shape[0],
                                    mpm_input.shape[1],
                                    self.mpm.args.max_length - prev_len,
                                )
                            ).to(mpm_input.device),
                        ],
                        dim=-1,
                    )
                mpm_output = self.mpm(
                    mpm_input.long(), return_layer=self.args.mpm_layer
                )
                out = mpm_output["representations"].detach().cpu()
                out = out[0, :prev_len, :]
                np.save(
                    Path(audio_path).with_suffix(f".mpm.npy"),
                    out,
                )
            else:
                out = np.load(file)
            results.append(out)

        # pad to max length
        if self.args.max_frames is None:
            max_len = np.max([r.shape[0] for r in results])
        else:
            max_len = self.args.max_frames

        mask = np.zeros((len(results), max_len))

        for i, r in enumerate(results):
            mask[i, : r.shape[0]] = 1

        batch["mask"] = torch.tensor(mask).bool()

        batch["mpm"] = torch.tensor(
            np.array([np.pad(r, ((0, max_len - r.shape[0]), (0, 0))) for r in results])
        ).to(torch.float32)

        batch["emotion"] = torch.tensor(batch["labels"]).long()
        batch["emotion_onehot"] = F.one_hot(
            batch["emotion"], num_classes=len(self.emotions2int)
        ).to(torch.float32)

        return batch


class BaselineTIMITCollator:
    def __init__(
        self,
        args: TIMITCollatorArgs,
        device: torch.device = None,
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
        if args.use_algorithmic_features:
            self.pitch_measure = PitchMeasure()
            self.energy_measure = EnergyMeasure()
            self.voice_activity_measure = VoiceActivityMeasure()

    def __call__(self, batch):
        batch = {k: [d[k] for d in batch] for k in batch[0]}
        results = {measure: [] for measure in self.args.measures.split(",")}
        phoneme_boundaries = []
        word_boundaries = []
        for k, audio in enumerate(batch["audio"]):
            audio_path = audio["path"]
            measures = {measure: [] for measure in ALL_MEASURES}
            original_len = len(audio["array"])
            for measure in ALL_MEASURES:
                file = Path(audio_path).with_suffix(f".{measure}.npy")
                if file.exists() and not self.args.overwrite:
                    measures[measure] = np.load(file)
                    vocex_len = len(measures[measure])
            if (not file.exists()) or self.args.overwrite:
                audio, sr = audio["array"], audio["sampling_rate"]
                if self.args.use_algorithmic_features:
                    # resample w to 22050
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
                    pitch = self.pitch_measure(audio, np.array([1000]))["measure"]
                    energy = self.energy_measure(audio, np.array([1000]))["measure"]
                    vad = self.voice_activity_measure(audio, np.array([1000]))[
                        "measure"
                    ]
                    if self.args.use_cwt:
                        pitch = (pitch - pitch.mean()) / (pitch.std() + 1e-8)
                        energy = (energy - energy.mean()) / (energy.std() + 1e-8)
                        vad = (vad - vad.mean()) / (vad.std() + 1e-8)
                        pitch = cwt(
                            pitch, ricker, np.arange(1, self.args.cwt_n_bins + 1)
                        ).T
                        energy = cwt(
                            energy, ricker, np.arange(1, self.args.cwt_n_bins + 1)
                        ).T
                        vad = cwt(vad, ricker, np.arange(1, self.args.cwt_n_bins + 1)).T
                    vocex_output = {
                        "measures": {
                            "pitch": torch.tensor(pitch),
                            "energy": torch.tensor(energy),
                            "voice_activity_binary": torch.tensor(vad),
                        }
                    }
                else:
                    vocex_output = self.vocex(audio, sr)
                for measure in ALL_MEASURES:
                    if not self.args.use_cwt:
                        v = vocex_output["measures"][measure]
                        v = v.flatten()
                        vocex_len = len(v)
                        # min-max normalize
                        if (v.max() - v.min()) == 0:
                            v = np.zeros_like(v)
                        else:
                            if measure == "pitch":
                                v = np.clip(
                                    v, self.args.pitch_min, self.args.pitch_max
                                ) / (self.args.pitch_max - self.args.pitch_min)
                            elif measure == "energy":
                                v = np.clip(
                                    v, self.args.energy_min, self.args.energy_max
                                ) / (self.args.energy_max - self.args.energy_min)
                            elif measure == "voice_activity_binary":
                                v = np.clip(v, self.args.vad_min, self.args.vad_max) / (
                                    self.args.vad_max - self.args.vad_min
                                )
                    else:
                        v = vocex_output["measures"][measure]
                        v = v.numpy()
                        vocex_len = len(v)
                    measures[measure] = v

                    np.save(
                        Path(audio_path).with_suffix(f".{measure}.npy"),
                        v,
                    )
            for measure in results:
                results[measure].append(measures[measure])

            # fold up measures such the the length is half the original length, but we have an additional dimension
            for measure in results:
                if not self.args.use_cwt:
                    results[measure][-1] = np.array(
                        [
                            np.array(
                                [
                                    results[measure][-1][i],
                                    results[measure][-1][i + 1],
                                ]
                            )
                            if i + 1 < len(results[measure][-1])
                            else np.array(
                                [
                                    results[measure][-1][i],
                                    results[measure][-1][i],
                                ]
                            )
                            for i in range(0, len(results[measure][-1]), 2)
                        ]
                    )
                else:
                    results[measure][-1] = np.array(
                        [
                            np.array(
                                [
                                    results[measure][-1][i],
                                    results[measure][-1][i + 1],
                                ]
                            )
                            if i + 1 < len(results[measure][-1])
                            else np.array(
                                [
                                    results[measure][-1][i],
                                    results[measure][-1][i],
                                ]
                            )
                            for i in range(0, len(results[measure][-1]), 2)
                        ]
                    )
                    results[measure][-1] = np.array(
                        [
                            np.array(
                                [
                                    interpolate(
                                        results[measure][-1][i, :, j],
                                        vocex_len,
                                    )
                                    for j in range(results[measure][-1].shape[-1])
                                ]
                            )
                            for i in range(results[measure][-1].shape[0])
                        ]
                    )

            vocex_len = np.ceil(vocex_len / 2).astype(int)

            # align phoneme boundaries with vocex output
            phoneme_stops = batch["phonetic_detail"][k]["stop"][:-1]
            phoneme_stops = np.round(np.array(phoneme_stops) / original_len * vocex_len)
            phoneme_boundaries.append(phoneme_stops)
            # convert to sequence of 0s and 1s
            phoneme_boundaries[-1] = np.array(
                [1 if i in phoneme_boundaries[-1] else 0 for i in range(vocex_len)]
            )

            # align word boundaries with vocex output
            word_stops = batch["word_detail"][k]["stop"]
            word_stops = [batch["word_detail"][k]["start"][0]] + word_stops
            word_stops = np.round(np.array(word_stops) / original_len * vocex_len)
            word_boundaries.append(word_stops)
            # convert to sequence of 0s and 1s
            word_boundaries[-1] = np.array(
                [1 if i in word_boundaries[-1] else 0 for i in range(vocex_len)]
            )

        # pad to max length
        first_measure = list(results.keys())[0]
        if self.args.max_frames is None:
            max_len = np.max([r.shape[0] for r in results[first_measure]])
        else:
            max_len = self.args.max_frames

        mask = np.zeros((len(results[first_measure]), max_len))

        for i, r in enumerate(results[first_measure]):
            mask[i, : r.shape[0]] = 1

        batch["mask"] = torch.tensor(mask).bool()

        batch["measures"] = {measure: results[measure] for measure in results}

        # pad measures
        for measure in results:
            print([r.shape for r in batch["measures"][measure]])
            batch["measures"][measure] = torch.tensor(
                np.array(
                    [
                        np.pad(r, ((0, max_len - r.shape[0]), (0, 0)))
                        for r in batch["measures"][measure]
                    ]
                )
            ).to(torch.float32)

        # pad boundaries
        phoneme_boundaries = [
            np.pad(b, (0, max_len - b.shape[0])) for b in phoneme_boundaries
        ]
        word_boundaries = [
            np.pad(b, (0, max_len - b.shape[0])) for b in word_boundaries
        ]

        batch["phoneme_boundaries"] = torch.tensor(phoneme_boundaries).float()
        batch["word_boundaries"] = torch.tensor(word_boundaries).float()

        return batch


class ProsodyModelTIMITCollator:
    def __init__(
        self,
        args: TIMITCollatorArgs,
        device: torch.device = None,
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
        self.mpm = MaskedProsodyModel.from_pretrained(self.args.mpm)
        if self.args.use_mpm_random: # Reset the pretrained model weights to random init
            print(f"E.g. weights before init:\n{self.mpm.state_dict()['transformer.layers.0.self_attn.out_proj.weight']}")
            self.mpm.apply(self.mpm._init_all_weights)
            print(f"Same weights after init:\n{self.mpm.state_dict()['transformer.layers.0.self_attn.out_proj.weight']}")
        self.bins = torch.linspace(0, 1, self.mpm.args.bins)
        if device is not None:
            self.mpm.to(device)
        self.device = device
        if args.use_algorithmic_features:
            self.pitch_measure = PitchMeasure()
            self.energy_measure = EnergyMeasure()
            self.voice_activity_measure = VoiceActivityMeasure()

    def __call__(self, batch):
        batch = {k: [d[k] for d in batch] for k in batch[0]}
        results = []
        phoneme_boundaries = []
        word_boundaries = []
        for k, audio in enumerate(batch["audio"]):
            audio_path = audio["path"]
            measures = {measure: [] for measure in ALL_MEASURES}
            original_len = len(audio["array"])
            file = Path(audio_path).with_suffix(f".mpm.npy")
            if (not file.exists()) or self.args.overwrite:
                audio, sr = audio["array"], audio["sampling_rate"]
                if not self.args.use_algorithmic_features:
                    vocex_output = self.vocex(audio, sr)
                    pitch = vocex_output["measures"]["pitch"]
                    energy = vocex_output["measures"]["energy"]
                    vad = vocex_output["measures"]["voice_activity_binary"]
                else:
                    # resample w to 22050
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
                    pitch = self.pitch_measure(audio, np.array([1000]))["measure"]
                    energy = self.energy_measure(audio, np.array([1000]))["measure"]
                    vad = self.voice_activity_measure(audio, np.array([1000]))[
                        "measure"
                    ]
                # normalize using pitch_min, pitch_max, energy_min, energy_max, vad_min, vad_max
                pitch = torch.clip(
                    torch.tensor(pitch).unsqueeze(0),
                    self.args.pitch_min,
                    self.args.pitch_max,
                ) / (self.args.pitch_max - self.args.pitch_min)
                energy = torch.clip(
                    torch.tensor(energy).unsqueeze(0),
                    self.args.energy_min,
                    self.args.energy_max,
                ) / (self.args.energy_max - self.args.energy_min)
                vad = torch.clip(
                    torch.tensor(vad).unsqueeze(0), self.args.vad_min, self.args.vad_max
                )
                # bucketize
                pitch = torch.bucketize(pitch, self.bins) + 2
                pitch[pitch == self.mpm.args.bins + 2] = self.mpm.args.bins + 1
                energy = torch.bucketize(energy, self.bins) + 2
                vad = torch.bucketize(vad, torch.linspace(0, 1, 2)) + 2
                mpm_input = torch.stack(
                    [
                        pitch,
                        energy,
                        vad,
                    ]
                ).transpose(0, 1)
                if self.device is not None:
                    mpm_input = mpm_input.to(self.device)
                prev_len = mpm_input.shape[-1]
                if mpm_input.shape[-1] < self.mpm.args.max_length:
                    mpm_input = torch.cat(
                        [
                            mpm_input,
                            torch.zeros(
                                (
                                    mpm_input.shape[0],
                                    mpm_input.shape[1],
                                    self.mpm.args.max_length - prev_len,
                                )
                            ).to(mpm_input.device),
                        ],
                        dim=-1,
                    )
                elif mpm_input.shape[-1] > self.mpm.args.max_length:
                    mpm_input = mpm_input[:, :, : self.mpm.args.max_length]
                mpm_output = self.mpm(
                    mpm_input.long(), return_layer=self.args.mpm_layer
                )
                out = mpm_output["representations"].detach().cpu()
                if prev_len < self.mpm.args.max_length:
                    out = out[0, :prev_len, :]
                else:
                    # pad
                    out = torch.cat(
                        [
                            out[0, :prev_len, :],
                            torch.zeros(
                                (
                                    prev_len - self.mpm.args.max_length,
                                    out.shape[-1],
                                )
                            ),
                        ],
                        dim=0,
                    )
                np.save(
                    Path(audio_path).with_suffix(f".mpm.npy"),
                    out,
                )
            else:
                out = np.load(file)
            results.append(out)

            # fold up results such the the length is half the original length, but we have an additional dimension

            results[-1] = np.array(
                [
                    np.concatenate(
                        [
                            results[-1][i],
                            results[-1][i + 1],
                        ]
                    )
                    if i + 1 < len(results[-1])
                    else np.concatenate(
                        [
                            results[-1][i],
                            results[-1][i],
                        ]
                    )
                    for i in range(0, len(results[-1]), 2)
                ]
            )

            vocex_len = len(results[-1])

            # align phoneme boundaries with vocex output
            phoneme_stops = batch["phonetic_detail"][k]["stop"][:-1]
            phoneme_stops = np.round(np.array(phoneme_stops) / original_len * vocex_len)
            phoneme_boundaries.append(phoneme_stops)
            # convert to sequence of 0s and 1s
            phoneme_boundaries[-1] = np.array(
                [1 if i in phoneme_boundaries[-1] else 0 for i in range(vocex_len)]
            )

            # align word boundaries with vocex output
            word_stops = batch["word_detail"][k]["stop"]
            word_stops = [batch["word_detail"][k]["start"][0]] + word_stops
            word_stops = np.round(np.array(word_stops) / original_len * vocex_len)
            word_boundaries.append(word_stops)
            # convert to sequence of 0s and 1s
            word_boundaries[-1] = np.array(
                [1 if i in word_boundaries[-1] else 0 for i in range(vocex_len)]
            )

        # pad to max length
        if self.args.max_frames is None:
            max_len = np.max([r.shape[0] for r in results])
        else:
            max_len = self.args.max_frames

        mask = np.zeros((len(results), max_len))

        for i, r in enumerate(results):
            mask[i, : r.shape[0]] = 1

        batch["mask"] = torch.tensor(mask).bool()

        batch["mpm"] = torch.tensor(
            np.array([np.pad(r, ((0, max_len - r.shape[0]), (0, 0))) for r in results])
        ).to(torch.float32)

        # pad boundaries
        phoneme_boundaries = [
            np.pad(b, (0, max_len - b.shape[0])) for b in phoneme_boundaries
        ]
        word_boundaries = [
            np.pad(b, (0, max_len - b.shape[0])) for b in word_boundaries
        ]

        batch["phoneme_boundaries"] = torch.tensor(phoneme_boundaries).float()
        batch["word_boundaries"] = torch.tensor(word_boundaries).float()

        return batch
