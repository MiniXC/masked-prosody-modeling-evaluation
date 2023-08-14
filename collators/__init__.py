import yaml

from .collators import BaselineBURNCollator
from configs.args import CollatorArgs
import torch


def get_collator(args: CollatorArgs):
    return {
        "default": BaselineBURNCollator,
    }[
        args.name
    ](args)
