import yaml

from .collators import BaselineBURNCollator, BaselineRAVDESSCollator
from configs.args import BURNCollatorArgs, RAVDESSCollatorArgs
import torch


def get_collator(args: BURNCollatorArgs or RAVDESSCollatorArgs):
    return {
        "default_burn": BaselineBURNCollator,
        "default_ravdess": BaselineRAVDESSCollator,
    }[args.name](args)
