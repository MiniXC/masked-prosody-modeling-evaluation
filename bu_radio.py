"""(BURN) Boston University Radio News Corpus."""

import os
from pathlib import Path

import datasets
import numpy as np

logger = datasets.logging.get_logger(__name__)

_PATH = os.environ.get("BURN_PATH", None)

_VERSION = "0.0.2"

_CITATION = """\
@article{ostendorf1995boston,
  title={The Boston University radio news corpus},
  author={Ostendorf, Mari and Price, Patti J and Shattuck-Hufnagel, Stefanie},
  journal={Linguistic Data Consortium},
  pages={1--19},
  year={1995}
}
"""

_DESCRIPTION = """\
The Boston University Radio Speech Corpus was collected primarily to support research in text-to-speech synthesis, particularly generation of prosodic patterns. The corpus consists of professionally read radio news data, including speech and accompanying annotations, suitable for speech and language research. 
"""

_URL = "https://catalog.ldc.upenn.edu/LDC96S36"


class BURNConfig(datasets.BuilderConfig):
    """BuilderConfig for BURN."""

    def __init__(self, sampling_rate=16000, hop_length=256, win_length=1024, **kwargs):
        """BuilderConfig for BURN.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(BURNConfig, self).__init__(**kwargs)

        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.seconds_per_frame = hop_length / sampling_rate
        
        if _PATH is None:
            raise ValueError("Please set the environment variable BURN_PATH to point to the BURN dataset directory.")
        
class BURN(datasets.GeneratorBasedBuilder):
    """BURN dataset."""

    BUILDER_CONFIGS = [
        BURNConfig(
            name="burn",
            version=datasets.Version(_VERSION, ""),
        ),
    ]

    def _info(self):
        features = {
            "speaker": datasets.Value("string"),
            "words": datasets.Sequence(datasets.Value("string")),
            "word_durations": datasets.Sequence(datasets.Value("int32")),
            "prominence": datasets.Sequence(datasets.Value("bool")),
            "break": datasets.Sequence(datasets.Value("bool")),
            "audio": datasets.Value("string"),
        }

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
            supervised_keys=["prominence", "break"],
            homepage="https://catalog.ldc.upenn.edu/LDC96S36",
            citation=_CITATION,
            task_templates=None,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name="train",
                gen_kwargs={
                    "speakers": ["f1a", "f3a", "m1b", "m2b", "m3b", "m4b"],
                }
            ),
            datasets.SplitGenerator(
                name="dev",
                gen_kwargs={
                    "speakers": [],
                }
            ),
        ]

    def _generate_example(self, file):
        words = []
        word_ts = []
        word_durations = []
        if not file.with_suffix(".ton").exists():
            return None
        if not file.with_suffix(".brk").exists():
            return None
        if not file.with_suffix(".wrd").exists():
            return None
        with open(file.with_suffix(".wrd"), "r") as f:
            lines = f.readlines()
            lines = [line for line in lines if line != "\n"]
            # get index of "#\n" line
            idx = lines.index("#\n")
            lines = lines[idx+1:]
            lines = [tuple(line.strip().split()) for line in lines]
            # remove lines with no word
            lines = [line for line in lines if len(line) == 3]
            word_ts = np.array([float(start) for start, _, _ in lines])
            words = [word for _, _, word in lines]
        prominence = np.zeros(len(words))
        boundary = np.zeros(len(words))
        if len(words) <= 1:
            return None
        with open(file.with_suffix(".ton"), "r") as f:
            lines = f.readlines()
            lines = [line for line in lines if line != "\n"]
            wrd_idx = 0
            idx = lines.index("#\n")
            lines = lines[idx+1:]
            lines = [tuple(line.strip().split()[:3]) for line in lines]
            # remove lines with no word
            lines = [line for line in lines if len(line) == 3]
            for start, _, accent in lines:
                # find word index
                while float(start) > word_ts[wrd_idx]:
                    wrd_idx += 1
                    if wrd_idx >= len(word_ts):
                        # warning
                        logger.warning(f"Word index {wrd_idx} out of bounds for file {file}")
                        return None
                if accent in ['H*', 'L*', 'L*+H', 'L+H*', 'H+', '!H*']:
                    prominence[wrd_idx] = 1
        with open(file.with_suffix(".brk"), "r") as f:
            lines = f.readlines()
            lines = [line for line in lines if line != "\n"]
            wrd_idx = 0
            idx = lines.index("#\n")
            lines = lines[idx+1:]
            lines = [tuple(line.strip().split()) for line in lines]
            if np.abs(len(lines) - len(words)) > 2:
                logger.warning(f"Word count mismatch for file {file}")
                return None
            for l in lines:
                if len(l) < 3:
                    continue
                score = l[2]
                start = float(l[0])
                # find word index, by finding the start value closest to word_ts
                wrd_idx = np.argmin(np.abs(word_ts - start))
                if "3" in score or "4" in score:
                    boundary[wrd_idx] = 1
        # compute word durations using self.config.seconds_per_frame
        word_diff = np.concatenate([[word_ts[0]], np.diff(word_ts)])
        word_durations = np.round(word_diff / self.config.seconds_per_frame).astype(np.int32)
        return {
            "words": words,
            "word_durations": word_durations,
            "prominence": prominence,
            "break": boundary,
            "audio": str(file),
        }

    def _generate_examples(self, speakers):
        print(_PATH)
        files = list(Path(_PATH).glob("**/*.sph"))
        print(files)
        speakers = [str(file).replace(_PATH, "").split("/")[1] for file in files]
        #speaker_list.extend([speaker] * len(speaker_sph_files))
        j = 0
        for i, file in enumerate(files):
            example = self._generate_example(file)
            if example is not None:
                example["speaker"] = speakers[i]
                yield j, example
                j += 1
