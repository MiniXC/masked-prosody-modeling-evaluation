import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import numpy as np

from configs.args import BURNCollatorArgs, RAVDESSCollatorArgs, TIMITCollatorArgs


def plot_baseline_burn_batch(batch, args: BURNCollatorArgs):
    """
    the batch cotains a dict of measures, each with shape (batch_size, max_len, args.values_per_word)
    we plot each measure as an image, as well as the original audio
    """

    batch_size = len(batch["audio"])
    measures = args.measures.split(",")
    fig, axs = plt.subplots(
        len(measures) + 1, batch_size, figsize=(batch_size * 3, len(measures) * 3)
    )
    audio_and_sr = [librosa.load(audio) for audio in batch["audio"]]
    audios = [a for a, _ in audio_and_sr]
    srs = [sr for _, sr in audio_and_sr]
    # pad audio to max length
    max_len = np.max([a.shape[0] for a in audios])
    audios = [
        np.pad(a, (0, max_len - a.shape[0])) if a.shape[0] < max_len else a
        for a in audios
    ]
    mels = [
        np.log(librosa.feature.melspectrogram(y=a, sr=sr) + 1e-6)
        for a, sr in zip(audios, srs)
    ]
    # plot, while making sure the sizes are the same for each measure
    # disable ticks
    for i in range(batch_size):
        for j, measure in enumerate(measures):
            axs[j, i].imshow(
                batch["measures"][measure][i].T, aspect="auto", interpolation="none"
            )
            axs[j, i].set_title(measure)
            axs[j, i].set_xticks([])
            axs[j, i].set_yticks([])
        axs[-1, i].imshow(mels[i], aspect="auto", origin="lower", interpolation="none")
        axs[-1, i].set_title("audio")
        axs[-1, i].set_xticks([])
        axs[-1, i].set_yticks([])

    plt.tight_layout()

    return fig


def plot_baseline_ravdess_batch(batch, args: RAVDESSCollatorArgs):
    batch_size = len(batch["audio"])
    measures = args.measures.split(",")
    fig, axs = plt.subplots(
        len(measures) + 1, batch_size, figsize=(batch_size * 3, len(measures) * 3)
    )
    audios = [a["array"] for a in batch["audio"]]
    srs = [a["sampling_rate"] for a in batch["audio"]]
    # pad audio to max length
    max_len = np.max([a.shape[0] for a in audios])
    audios = [
        np.pad(a, (0, max_len - a.shape[0])) if a.shape[0] < max_len else a
        for a in audios
    ]
    mels = [
        np.log(librosa.feature.melspectrogram(y=a, sr=sr) + 1e-6)
        for a, sr in zip(audios, srs)
    ]
    # plot, while making sure the sizes are the same for each measure
    # disable ticks
    for i in range(batch_size):
        for j, measure in enumerate(measures):
            sns.lineplot(
                x=range(len(batch["measures"][measure][i])),
                y=batch["measures"][measure][i],
                ax=axs[j, i],
            )
            axs[j, i].set_title(measure)
            axs[j, i].set_xticks([])
            axs[j, i].set_yticks([])
        axs[-1, i].imshow(mels[i], aspect="auto", origin="lower", interpolation="none")
        axs[-1, i].set_title("audio")
        axs[-1, i].set_xticks([])
        axs[-1, i].set_yticks([])
    plt.tight_layout()

    return fig


def plot_baseline_timit_batch(batch, args: TIMITCollatorArgs):
    batch_size = len(batch["audio"])

    fig, axs = plt.subplots(batch_size, 2, figsize=(3, batch_size * 3))
    audios = [a["array"] for a in batch["audio"]]
    srs = [a["sampling_rate"] for a in batch["audio"]]
    # resample to 22050
    audios = [
        librosa.resample(y=a, orig_sr=sr, target_sr=22050) for a, sr in zip(audios, srs)
    ]
    srs = [22050 for _ in srs]
    mels = [
        np.log(
            librosa.feature.melspectrogram(
                y=a, sr=sr, n_fft=2048, hop_length=512, win_length=2048
            )
            + 1e-6
        )
        for a, sr in zip(audios, srs)
    ]
    max_len = args.max_frames
    # pad mels to max length
    mels = [
        np.pad(m, ((0, 0), (0, max_len - m.shape[1])), mode="constant")
        if m.shape[1] < max_len
        else m
        for m in mels
    ]
    # plot, while making sure the sizes are the same for each measure
    # disable ticks
    for i in range(batch_size):
        axs[i, 0].imshow(mels[i], aspect="auto", origin="lower", interpolation="none")
        axs[i, 0].set_title("audio")
        axs[i, 0].set_xticks([])
        axs[i, 0].set_yticks([])
        sns.lineplot(
            x=range(len(batch["phoneme_boundaries"][i])),
            y=batch["phoneme_boundaries"][i] * 120,
            ax=axs[i, 0],
        )
        sns.lineplot(
            x=range(len(batch["word_boundaries"][i])),
            y=batch["word_boundaries"][i] * 120,
            ax=axs[i, 0],
        )
        sns.lineplot(
            x=range(len(batch["measures"]["pitch"][i])),
            y=batch["measures"]["pitch"][i].mean(axis=-1),
            ax=axs[i, 1],
        )
        sns.lineplot(
            x=range(len(batch["measures"]["energy"][i])),
            y=batch["measures"]["energy"][i].mean(axis=-1),
            ax=axs[i, 1],
        )
        sns.lineplot(
            x=range(len(batch["measures"]["voice_activity_binary"][i])),
            y=batch["measures"]["voice_activity_binary"][i].mean(axis=-1),
            ax=axs[i, 1],
        )
        axs[i, 1].set_title("measures")

    plt.tight_layout()

    return fig
