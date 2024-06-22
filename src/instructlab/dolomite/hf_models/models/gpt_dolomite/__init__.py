# ----------------------------------------------------------------
# Extracted from https://github.com/ibm-granite/dolomite-engine
# ----------------------------------------------------------------
# Local
from .base import GPTDolomiteModel, GPTDolomitePreTrainedModel
from .main import GPTDolomiteForCausalLM
from .mlp import interleave_up_gate_tensor_for_mlp, split_up_gate_tensor_for_mlp
