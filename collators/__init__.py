from typing import Dict, List, Tuple, Union

import yaml

from .collators import (
    BaselineBURNCollator,
    BaselineRAVDESSCollator,
    BaselineTIMITCollator,
    ProsodyModelBURNCollator,
    ProsodyModelRAVDESSCollator,
    ProsodyModelTIMITCollator,
    ProsodyModelSWBCollator
)
from configs.args import BURNCollatorArgs, RAVDESSCollatorArgs, SWBCollatorArgs, TIMITCollatorArgs


def get_collator(
    args: Union[TIMITCollatorArgs, BURNCollatorArgs, SWBCollatorArgs, RAVDESSCollatorArgs], device=None
):
    return {
        "default_burn": BaselineBURNCollator,
        "default_ravdess": BaselineRAVDESSCollator,
        # "default_swb": BaselineSWBCollator,
        "default_timit": BaselineTIMITCollator,
        "prosody_model_burn": ProsodyModelBURNCollator,
        "prosody_model_ravdess": ProsodyModelRAVDESSCollator,
        "prosody_model_swb": ProsodyModelSWBCollator,
        "prosody_model_timit": ProsodyModelTIMITCollator,
    }[args.name](args, device=device)
