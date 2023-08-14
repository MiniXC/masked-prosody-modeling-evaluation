from typing import Dict, List, Tuple, Union

import yaml

from .collators import (
    BaselineBURNCollator,
    BaselineRAVDESSCollator,
    BaselineTIMITCollator,
)
from configs.args import BURNCollatorArgs, RAVDESSCollatorArgs, TIMITCollatorArgs
import torch


def get_collator(args: Union[TIMITCollatorArgs, BURNCollatorArgs, RAVDESSCollatorArgs]):
    return {
        "default_burn": BaselineBURNCollator,
        "default_ravdess": BaselineRAVDESSCollator,
        "default_timit": BaselineTIMITCollator,
    }[args.name](args)
