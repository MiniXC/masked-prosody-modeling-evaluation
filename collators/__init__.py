from typing import Dict, List, Tuple, Union

import yaml

from .collators import (
    BaselineBURNCollator,
    BaselineRAVDESSCollator,
    BaselineTIMITCollator,
    ProsodyModelBURNCollator,
    ProsodyModelRAVDESSCollator,
    ProsodyModelTIMITCollator,
)
from configs.args import BURNCollatorArgs, RAVDESSCollatorArgs, TIMITCollatorArgs


def get_collator(args: Union[TIMITCollatorArgs, BURNCollatorArgs, RAVDESSCollatorArgs], device = None):
    return {
        "default_burn": BaselineBURNCollator,
        "default_ravdess": BaselineRAVDESSCollator,
        "default_timit": BaselineTIMITCollator,
        "prosody_model_burn": ProsodyModelBURNCollator,
        "prosody_model_ravdess": ProsodyModelRAVDESSCollator,
        "prosody_model_timit": ProsodyModelTIMITCollator,
    }[args.name](args, device=device)