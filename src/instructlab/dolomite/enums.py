# ----------------------------------------------------------------
# Extracted from https://github.com/ibm-granite/dolomite-engine
# ----------------------------------------------------------------
# Standard
from enum import Enum


class ParamsGroupMethod(Enum):
    mup = "mup"


class GradientCheckpointingMethod(Enum):
    block = "block"
