from typing import Tuple

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from ...config import CommonConfig
from ...enums import AttentionHeadType, PositionEmbeddingType
from ..linear import Linear
from ..position_embedding import apply_rotary_pos_emb
from .utils import repeat_key_value


class Attention(torch.nn.Module):
    def __init__(
        self, config: CommonConfig, causal: bool, layer_idx: int = None
    ) -> None:
        super().__init__()

        self.causal = causal
        self.hidden_size = config.n_embd
        self.num_heads = config.n_head
        self.num_key_value_heads = config.num_key_value_heads
        self.add_bias = config.add_bias

        assert (
            self.hidden_size % self.num_heads == 0
        ), f"`hidden_size` ({self.hidden_size}) must be divisible by `num_heads` ({self.num_heads})"

        self.head_dim = self.hidden_size // self.num_heads
        self.attention_head_type = AttentionHeadType(config.attention_head_type)

        self.position_embedding_type = PositionEmbeddingType(
            config.position_embedding_type
        )
        self.scale_attn_weights = config.scale_attn_weights
        self.attention_multiplier = config.attention_multiplier

        self.layer_idx = layer_idx
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = (
            config.scale_attention_softmax_in_fp32 and config.attention_softmax_in_fp32
        )

        if self.attention_head_type == AttentionHeadType.mha:
            if self.num_key_value_heads is None:
                self.num_key_value_heads = self.num_heads

            assert (
                self.num_heads == self.num_key_value_heads
            ), f"{self.__class__.__name__} should have same number of heads for query, keys and values"
        elif self.attention_head_type == AttentionHeadType.gqa:
            assert (
                self.num_key_value_heads is not None
            ), "`num_key_value_heads` needs to be specified with GroupedQueryAttention"

            assert self.num_heads % self.num_key_value_heads == 0, (
                f"`num_heads` ({self.num_heads}) should be a multiple of `num_key_value_heads` "
                f"({self.num_key_value_heads})"
            )
        elif self.attention_head_type == AttentionHeadType.mqa:
            if self.num_key_value_heads is None:
                self.num_key_value_heads = 1

            assert (
                self.num_key_value_heads == 1
            ), f"{self.__class__.__name__} should have 1 head for keys and values"
        else:
            raise ValueError(
                f"unexpected attention_head_type ({self.attention_head_type})"
            )

        # note that the actual layout is different for the output and depends on whether we are using MHA, MQA or GQA
        # (self.hidden_size + 2 * self.num_key_value_heads * self.head_dim) is just the actual number output features
        self.c_attn = Linear(
            self.hidden_size,
            self.hidden_size + 2 * self.num_key_value_heads * self.head_dim,
            bias=self.add_bias,
        )

        self.c_proj = Linear(self.hidden_size, self.hidden_size, bias=self.add_bias)

        self.attn_pdrop = config.attn_pdrop
        self.resid_pdrop = config.resid_pdrop

        self.attn_dropout = (
            torch.nn.Identity() if self.attn_pdrop == 0 else torch.nn.Dropout(self.attn_pdrop)
        )
        self.resid_dropout = (
            torch.nn.Identity() if self.resid_pdrop == 0 else torch.nn.Dropout(self.resid_pdrop)
        )

    def _prepare_qkv_for_forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        # the output of following is a tuple if using MQA with tensor parallel
        hidden_states = self.c_attn(hidden_states)

        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, [num_heads + num_key_value_heads * 2] * head_dim)
        # ==========================================================================================

        # for MHA, we can get away with doing just 1 transpose which is not true for GQA
        if self.attention_head_type == AttentionHeadType.mha:
            query, key, value = self._prepare_qkv_for_forward_mha(hidden_states)
        elif self.attention_head_type == AttentionHeadType.gqa:
            query, key, value = self._prepare_qkv_for_forward_gqa(hidden_states)
        elif self.attention_head_type == AttentionHeadType.mqa:
            query, key, value = self._prepare_qkv_for_forward_mqa(hidden_states)
        else:
            raise ValueError(
                f"unexpected attention_head_type ({self.attention_head_type})"
            )

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, query_length, head_dim)
        # value -> (batch_size, num_key_value_heads, query_length, head_dim)
        # ==========================================================================================

        return query, key, value

    def _prepare_qkv_for_forward_mha(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length = hidden_states.shape[:-1]

        hidden_states = hidden_states.view(batch_size, query_length, self.num_heads, -1)
        hidden_states = hidden_states.transpose(1, 2)

        query, key, value = hidden_states.chunk(3, dim=-1)

        return query, key, value

    def _prepare_qkv_for_forward_gqa(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length = hidden_states.shape[:-1]

        hidden_states = hidden_states.view(
            batch_size, query_length, self.num_key_value_heads, -1
        )

        query, key, value = hidden_states.split(
            (
                (self.num_heads // self.num_key_value_heads) * self.head_dim,
                self.head_dim,
                self.head_dim,
            ),
            dim=-1,
        )

        # this needs to be a reshape instead of view sadly
        query = query.reshape(batch_size, query_length, -1, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        return query, key, value

    def _prepare_qkv_for_forward_mqa(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, query_length = hidden_states.shape[:-1]

        query, key, value = hidden_states.split(
            (self.hidden_size, self.head_dim, self.head_dim), dim=-1
        )

        query = query.view(batch_size, query_length, self.num_heads, -1)

        query = query.transpose(1, 2)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)

        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache = None,
        attention_mask: torch.Tensor = None,
        rope_cos_sin: torch.Tensor = None,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: torch.Tensor = None,
    ) -> torch.Tensor:
        # ==========================================================================================
        # hidden_states -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        query, key, value = self._prepare_qkv_for_forward(hidden_states)

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, query_length, head_dim)
        # value -> (batch_size, num_key_value_heads, query_length, head_dim)
        # ==========================================================================================

        if self.position_embedding_type == PositionEmbeddingType.rope:
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, self.layer_idx)

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, key_length, head_dim)
        # value -> (batch_size, num_key_value_heads, key_length, head_dim)
        # ==========================================================================================

        key = key.transpose(-1, -2)

        dtype = query.dtype
        softmax_dtype = torch.float32 if self.attention_softmax_in_fp32 else dtype

        # ==========================================================================================
        # query -> (batch_size, num_heads, query_length, head_dim)
        # key -> (batch_size, num_key_value_heads, head_dim, key_length)
        # value -> (batch_size, num_key_value_heads, key_length, head_dim)
        # ==========================================================================================

        batch_size = query.shape[0]
        query_length = query.shape[2]
        key_length = key.shape[-1]

        key = repeat_key_value(key, self.num_heads, self.num_key_value_heads)
        value = repeat_key_value(value, self.num_heads, self.num_key_value_heads)

        # Always copies
        query = query.reshape(batch_size * self.num_heads, query_length, self.head_dim)
        # No copy when layer_past is provided.
        key = key.reshape(batch_size * self.num_heads, self.head_dim, key_length)

        # ==========================================================================================
        # query -> (batch_size * num_heads, query_length, head_dim)
        # key -> (batch_size * num_heads, head_dim, key_length)
        # value -> (batch_size, num_heads, key_length, head_dim)
        # ==========================================================================================

        if attention_mask is None:
            attn_weights = torch.empty(
                (batch_size * self.num_heads, query_length, key_length),
                device=query.device,
                dtype=query.dtype,
            )
            beta = 0
        else:
            attn_weights = attention_mask.expand(-1, self.num_heads, -1, -1).reshape(
                -1, query_length, key_length
            )
            beta = 1

        attn_weights = torch.baddbmm(
            attn_weights, query, key, beta=beta, alpha=self._get_softmax_scale(False)
        ).view(batch_size, self.num_heads, query_length, key_length)

        # ==========================================================================================
        # attn_weights -> (batch_size, num_heads, query_length, key_length)
        # ==========================================================================================

        attn_weights = F.softmax(attn_weights.to(softmax_dtype), dim=-1).to(dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # ==========================================================================================
        # value -> (batch_size, num_heads, key_length, head_dim)
        # attn_weights -> (batch_size, num_heads, query_length, key_length)
        # ==========================================================================================

        attn_output = torch.matmul(attn_weights, value)

        # ==========================================================================================
        # attn_output -> (batch_size, num_heads, query_length, head_dim)
        # ==========================================================================================

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(
            batch_size, -1, self.num_heads * self.head_dim
        )

        # ==========================================================================================
        # attn_output -> (batch_size, query_length, num_heads * head_dim)
        # ==========================================================================================

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output

    def _get_softmax_scale(self, return_none_allowed: bool = True) -> float:
        if self.scale_attn_weights:
            if self.attention_multiplier is None:
                softmax_scale = None if return_none_allowed else 1 / self.head_dim**0.5
            else:
                softmax_scale = self.attention_multiplier
        else:
            softmax_scale = 1

        return softmax_scale
