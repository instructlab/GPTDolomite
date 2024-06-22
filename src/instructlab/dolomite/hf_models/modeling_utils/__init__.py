# ----------------------------------------------------------------
# Extracted from https://github.com/ibm-granite/dolomite-engine
# ----------------------------------------------------------------
# Local
from .activations import get_activation_function, is_glu
from .attention import (
    SDPA,
    Attention,
    FlashAttention2,
    PaddingFreeAttention,
    get_attention_module,
    get_unpad_data,
    interleave_query_key_value_tensor_for_attention,
    repeat_key_value,
    split_query_key_value_tensor_for_attention,
)
from .normalization import RMSNorm, get_normalization_function
from .position_embedding import Alibi, RoPE, apply_rotary_pos_emb
