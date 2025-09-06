from typing import List, Optional, Tuple, Union
import math

from dataclasses import dataclass

from torch import nn
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
import numpy as np
import time

import transformers
from transformers import GenerationConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import ModelOutput, BaseModelOutputWithPast
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.utils import (
    logging,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

from .conversation import get_conv_template
from .configuration_onecat import (
    OneCatVLChatConfig,
    PatchEmbeddingConfig
)
from .var_model.infinity.models.infinity import sample_with_top_k_top_p_also_inplace_modifying_logits_
from .var_model.infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
from .peft.tuners.lora import Linear


logger = logging.get_logger(__name__)
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    from torch.nn.attention.flex_attention import or_masks, and_masks
    flex_attention = torch.compile(flex_attention)
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 4096
except:
    print('enable flex attetnion error')

_CONFIG_FOR_DOC = "Qwen2Config"


logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


def _make_causal_mask_i2t(
    input_ids_shape: torch.Size, 
    dtype: torch.dtype, 
    device: torch.device, 
    past_key_values_length: int = 0, 
    ue_token_mask : Optional[torch.Tensor] = None,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask_list=[]
    for i in range(bsz):
        mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        if ue_token_mask is not None:
            ue_token_mask_bool = ue_token_mask[i].flatten().bool()  # (tgt_len,)
            visual_rows = ue_token_mask_bool.unsqueeze(-1)  # (tgt_len, 1)
            visual_cols = ue_token_mask_bool.unsqueeze(0)   # (1, tgt_len)
            visual_mask_2d = visual_rows & visual_cols  # (tgt_len, tgt_len)
            mask = torch.where(visual_mask_2d, 0, mask)
        mask = mask.to(dtype)
        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
        mask_list.append(mask)
    mask = torch.stack(mask_list,dim=0)
    
    return mask[:, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _make_causal_mask_t2i(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    ue_token_mask: Optional[torch.Tensor] = None,
    ge_token_mask: Optional[torch.Tensor] = None,
    image_gen_decoding: Optional[bool] = None,
    scale_schedule: Optional[list] = None
) :

    bsz, tgt_len = input_ids_shape


    if image_gen_decoding: # for inference
        index = tgt_len
        ge_token_mask_1d_list = torch.full((bsz, tgt_len), torch.tensor(index, device=device), device=device)
        mask = None
        return mask,ge_token_mask_1d_list

    mask = torch.full((bsz,tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

    if ge_token_mask is not None or ue_token_mask is not None:
        ge_token_mask_2d_list = []
        ge_token_mask_1d_list = []
        ue_token_mask_2d_list = []
        for j in range(bsz):
            if ue_token_mask is not None: # for editing
                ue_token_mask_bool = ue_token_mask[j].flatten().bool()  # (tgt_len,)
                ue_token_mask_rows = ue_token_mask_bool.unsqueeze(-1)  # (tgt_len, 1)
                ue_token_mask_cols = ue_token_mask_bool.unsqueeze(0)   # (1, tgt_len)
                ue_token_mask_2d = ue_token_mask_rows & ue_token_mask_cols  # (tgt_len, tgt_len)
                ue_token_mask_2d_list.append(ue_token_mask_2d)
            if ge_token_mask is not None:
                ge_token_mask = ge_token_mask[j].squeeze(-1).clone()
                patch_nums = scale_schedule
                current_idx = (ge_token_mask == 1).nonzero(as_tuple=True)[0][0].item()
                for idx,num in enumerate(patch_nums):  # num (t,h,w) :(1,1,1), (1,2,2), (1,4,4), (1,6,6) ,(1,8,8), (1,12,12), (1,16,16) ....
                    block_size = num[-1]*num[-2]
                    end_idx = current_idx + block_size
                    ge_token_mask[current_idx:end_idx] = idx + 1
                    current_idx = end_idx

                ge_token_mask_rows = ge_token_mask.unsqueeze(-1)
                ge_token_mask_cols = ge_token_mask.unsqueeze(0)
                ge_token_mask_2d = (ge_token_mask_rows>=ge_token_mask_cols)&((ge_token_mask_rows+ge_token_mask_cols)>0)
                ge_token_mask_2d_list.append(ge_token_mask_2d)
                ge_token_mask_1d_list.append(ge_token_mask)
        if ue_token_mask is not None and ge_token_mask is not None:
            ue_token_mask_2d_list = torch.stack(ue_token_mask_2d_list)
            ge_token_mask_2d_list = torch.stack(ge_token_mask_2d_list)
            ge_token_mask_1d_list = torch.stack(ge_token_mask_1d_list)
            mask = torch.where(ge_token_mask_2d_list|ue_token_mask_2d_list, 0, mask)
        elif ge_token_mask is not None:
            ge_token_mask_2d_list = torch.stack(ge_token_mask_2d_list)
            ge_token_mask_1d_list = torch.stack(ge_token_mask_1d_list)
            mask = torch.where(ge_token_mask_2d_list, 0, mask)
        elif ue_token_mask is not None:
            ue_token_mask_2d_list = torch.stack(ue_token_mask_2d_list)
            mask = torch.where(ue_token_mask_2d_list, 0, mask)
    else:
        ge_token_mask_1d_list = torch.full((bsz, tgt_len), torch.tensor(0, device=device), device=device)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(bsz,tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[:, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length),ge_token_mask_1d_list


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2RotaryEmbeddingI2T(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


# Copied from transformers.models.mistral.modeling_mistral.MistralRotaryEmbedding with Mistral->Qwen2
class Qwen2RotaryEmbeddingT2I(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)



    def forward(self, x, position_ids,seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(2, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
    
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype),sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb_i2t(q, k, cos, sin, position_ids, unsqueeze_dim=1):

    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _apply_rotary_pos_emb_t2i(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    mrope_section = 2
    cos = torch.cat([m[i % 2] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 2] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x,multi_scale_ge_lora_mask=None):
        if multi_scale_ge_lora_mask is None:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        else:
            return self.down_proj(self.act_fn(self.gate_proj(x,multi_scale_ge_lora_mask)) * self.up_proj(x,multi_scale_ge_lora_mask),multi_scale_ge_lora_mask)


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb_i2t = Qwen2RotaryEmbeddingI2T(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.rotary_emb_t2i =Qwen2RotaryEmbeddingT2I(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.config.rope_theta
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            task = None,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        if task =='i2t':# i2t rope
            cos, sin = self.rotary_emb_i2t(value_states, seq_len=kv_seq_len)
            query_states, key_states = _apply_rotary_pos_emb_i2t(query_states, key_states, cos, sin, position_ids)
        elif task =='t2i': # t2i rope  
            cos, sin = self.rotary_emb_t2i(value_states,position_ids, seq_len=kv_seq_len)
            query_states, key_states = _apply_rotary_pos_emb_t2i(query_states, key_states, cos, sin, position_ids)
       

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
    

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def forward_flex(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        task = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        if task =='i2t':# i2t rope
            cos, sin = self.rotary_emb_i2t(value_states, seq_len=kv_seq_len)
            query_states, key_states = _apply_rotary_pos_emb_i2t(query_states, key_states, cos, sin, position_ids)
        else: # t2i rope  
            cos, sin = self.rotary_emb_t2i(value_states,position_ids, seq_len=kv_seq_len)
            query_states, key_states = _apply_rotary_pos_emb_t2i(query_states, key_states, cos, sin, position_ids)
    
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        if attention_mask is not None:
            attention_mask = attention_mask.to(query_states.device)  # Move to same device


        attn_output = flex_attention(query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), block_mask=attention_mask, enable_gqa=True).transpose(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2Attention,
}


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = Qwen2Attention(config, layer_idx)
        self.config = config
        self.mlp_te = Qwen2MLP(config)
        self.mlp_ue = Qwen2MLP(config)
        self.mlp_ge = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, 
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            image_gen_decoding: Optional[bool] = None,
            ue_token_mask: Optional[torch.Tensor] = None,
            ge_token_mask: Optional[torch.Tensor] = None,
            multi_scale_ge_lora_mask : Optional[torch.Tensor] = None,
        ):

        if multi_scale_ge_lora_mask is not None or ge_token_mask is not None or image_gen_decoding is not None:
            return self._forward_t2i(hidden_states,attention_mask,position_ids,past_key_value,output_attentions,use_cache,image_gen_decoding,ue_token_mask,ge_token_mask,multi_scale_ge_lora_mask)
        else:
            return self._forward_i2t(hidden_states,attention_mask,position_ids,past_key_value,output_attentions,use_cache,ue_token_mask)

    def _forward_i2t(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            ue_token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
    
        assert ue_token_mask is not None or past_key_value is not None
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        if self.config.use_flex_for_i2t:
        # Self Attention
            hidden_states, self_attn_weights, present_key_value = self.self_attn.forward_flex(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                task = 'i2t',
                use_cache=use_cache,
            )
        else:            
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                task = 'i2t',
                use_cache=use_cache,
            )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if past_key_value is None or ue_token_mask.shape[1] == hidden_states.shape[1]:
            if self.training:
                hidden_states = self.mlp_te(hidden_states)*(1.-ue_token_mask)+ self.mlp_ue(hidden_states)*ue_token_mask
            else:
                dim=hidden_states.shape[-1]
                ue_token_mask=ue_token_mask.repeat(1,1,dim).bool()
                non_ue_token_mask=~ue_token_mask
                if ue_token_mask.any():
                    hidden_states[ue_token_mask] = self.mlp_ue(hidden_states[ue_token_mask].reshape(-1,dim)).reshape(-1)
                if (non_ue_token_mask).any(): 
                    hidden_states[non_ue_token_mask] = self.mlp_te(hidden_states[non_ue_token_mask].reshape(-1,dim)).reshape(-1)
        else:
            hidden_states = self.mlp_te(hidden_states)

        
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
    def _forward_t2i(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            image_gen_decoding: Optional[bool] = False,
            ue_token_mask: Optional[torch.Tensor] = None,
            ge_token_mask: Optional[torch.Tensor] = None,
            multi_scale_ge_lora_mask : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        # assert ue_token_mask is not None or past_key_value is not None
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if self.config.use_flex_for_t2i:
            hidden_states, self_attn_weights, present_key_value = self.self_attn.forward_flex(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                task='t2i',
                use_cache=use_cache)
        else:
            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                task='t2i',
                use_cache=use_cache
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
            
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        if image_gen_decoding: 
            hidden_states = self.mlp_ge(hidden_states,multi_scale_ge_lora_mask)
            
        else:
            ge_token_mask_bool = ge_token_mask.bool() if ge_token_mask is not None else None
            ue_token_mask_bool = ue_token_mask.bool() if ue_token_mask is not None else None
            text_mask_bool = None
            if ge_token_mask_bool is not None:
                text_mask_bool = ~ge_token_mask_bool
                if ue_token_mask_bool is not None:
                    text_mask_bool = text_mask_bool & ~ue_token_mask_bool
            elif ue_token_mask_bool is not None:
                text_mask_bool = ~ue_token_mask_bool
            else:
                text_mask_bool = torch.ones_like(hidden_states[..., 0], dtype=torch.bool).unsqueeze(-1)
        
            hidden_states_temp = torch.zeros_like(hidden_states, device=device, dtype=dtype)
            if ue_token_mask_bool is not None:
                ue_tokens = hidden_states[ue_token_mask_bool.expand_as(hidden_states)].view(batch_size,-1, hidden_dim)
                processed_ue = self.mlp_ue(ue_tokens)
                hidden_states_temp.masked_scatter_(ue_token_mask_bool, processed_ue)

            if ge_token_mask_bool  is not None:
                ge_tokens = hidden_states[ge_token_mask_bool.expand_as(hidden_states)].view(batch_size,-1, hidden_dim)
                multi_scale_ge_lora_mask = [mask[ge_token_mask_bool.squeeze(0).squeeze(-1)>0] for mask in multi_scale_ge_lora_mask]
                processed_ge = self.mlp_ge(ge_tokens,multi_scale_ge_lora_mask)
                hidden_states_temp.masked_scatter_(ge_token_mask_bool, processed_ge)

            if text_mask_bool is not None:
                text_tokens = hidden_states[text_mask_bool.expand_as(hidden_states)].view(batch_size,-1, hidden_dim)
                processed_text = self.mlp_te(text_tokens)
                hidden_states_temp.masked_scatter_(text_mask_bool, processed_text)

            hidden_states = hidden_states_temp
            

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


QWEN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


QWEN2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        print('self._attn_implementation',self._attn_implementation)
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    def _prepare_decoder_attention_mask_i2t(self, attention_mask, input_shape, inputs_embeds, past_key_values_length, ue_token_mask=None):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask_i2t(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
                ue_token_mask=ue_token_mask,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    
    def _prepare_decoder_flex_attention_mask_i2t(self,  inputs_embeds,attention_mask,ue_token_mask=None):

        attention_mask = attention_mask.squeeze(0)*1
        if inputs_embeds.shape[0]==1:
            attention_mask = attention_mask.unsqueeze(0).clone()

        if ue_token_mask is not None: 

            ue_token_mask = ue_token_mask.squeeze(-1).clone()

      
        def prefix_mllm_causal_mask(ue_token_mask_temp,attention_mask):

            def causal_mask(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx
            
            def ue_token_mask(b, h, q_idx, kv_idx):
                return (ue_token_mask_temp[b,q_idx] >0) & (ue_token_mask_temp[b,kv_idx] >0)

            def padding_mask(b, h, q_idx, kv_idx):
                return (attention_mask[b,q_idx]>0) & (attention_mask[b,kv_idx]>0)

            return and_masks(or_masks(causal_mask,ue_token_mask),padding_mask)


        q_len = inputs_embeds.shape[1]
        bs = inputs_embeds.shape[0]
        sparse_mask = prefix_mllm_causal_mask(ue_token_mask.to(attention_mask),attention_mask)
        block_mask = create_block_mask(sparse_mask,
                B = bs, H =  self.config.num_attention_heads, 
                Q_LEN = q_len, KV_LEN = q_len, device = 'cuda',
                BLOCK_SIZE=128,_compile = True)

        return block_mask

    def forward(self, *args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if (
            'ge_token_mask' in kwargs
            or 'image_gen_decoding' in kwargs
        ):
            return self._forward_t2i( *args, **kwargs)
        else:
            return self._forward_i2t(*args, **kwargs)
        
    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def _forward_i2t(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            ue_token_mask: Optional[torch.Tensor] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)


        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)


        if self.training and self.config.use_flex_for_i2t:
            attention_mask = self._prepare_decoder_flex_attention_mask_i2t(
                        inputs_embeds,attention_mask,ue_token_mask
                )
        else:
            attention_mask = self._prepare_decoder_attention_mask_i2t(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length,ue_token_mask
                )
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    output_attentions,
                    use_cache,
                    None,
                    ue_token_mask,

                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    ue_token_mask=ue_token_mask,
                )
            
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )



    def _prepare_decoder_flex_attention_mask_t2i(self,  inputs_embeds,attention_mask,ue_token_mask=None,ge_token_mask=None,scale_schedule =None):
        attention_mask = attention_mask.squeeze(0)*1
        bs = inputs_embeds.shape[0]
        if inputs_embeds.shape[0]==1:
            attention_mask = attention_mask.unsqueeze(0).clone()

        if ue_token_mask is not None: 
            ue_token_mask = ue_token_mask.squeeze(-1).clone()

        if ge_token_mask is not None: 
            ge_token_mask = ge_token_mask.squeeze(-1).clone()      
            ge_token_mask_1d_list = []
            for j in range(bs):
                ge_token_mask_per_sample = ge_token_mask[j].clone()
                current_idx = (ge_token_mask_per_sample == 1).nonzero(as_tuple=True)[0][0].item()
                for num in scale_schedule:  # num (t,h,w) :(1,1,1), (1,2,2), (1,4,4), (1,6,6) ,(1,8,8), (1,12,12), (1,16,16)
                    block_size = num[-1]*num[-2]
                    end_idx = current_idx + block_size
                    ge_token_mask_per_sample[current_idx:end_idx] = block_size
                    current_idx = end_idx
                ge_token_mask_1d_list.append(ge_token_mask_per_sample)
            ge_token_mask_1d_list = torch.stack(ge_token_mask_1d_list)

        def prefix_mllm_causal_mask(ge_token_mask_1d_list,ge_token_mask_temp,ue_token_mask_temp,attention_mask):

            def causal_mask(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx
            
            def ge_token_var_mask(b, h, q_idx, kv_idx):
                return (ge_token_mask_1d_list[b,q_idx]>= ge_token_mask_1d_list[b,kv_idx]) & ((ge_token_mask_temp[b,q_idx]+ge_token_mask_temp[b,kv_idx])>0)

            def ue_token_mask(b, h, q_idx, kv_idx):
                return (ue_token_mask_temp[b,q_idx] >0) & (ue_token_mask_temp[b,kv_idx] >0)

            def padding_mask(b, h, q_idx, kv_idx):
                return (attention_mask[b,q_idx]>0) & (attention_mask[b,kv_idx]>0)
            
            if ue_token_mask_temp is None:
                return and_masks(or_masks(causal_mask,ge_token_var_mask),padding_mask)
            else:
                return and_masks(or_masks(causal_mask,ge_token_var_mask,ue_token_mask),padding_mask)
        
        sparse_mask = prefix_mllm_causal_mask(ge_token_mask_1d_list.to(attention_mask.dtype),ge_token_mask.to(attention_mask.dtype),attention_mask)

    
        q_len = inputs_embeds.shape[1]
        bs = inputs_embeds.shape[0]
        block_mask = create_block_mask(sparse_mask,
                B = bs, H =  self.config.num_attention_heads, 
                Q_LEN = q_len, KV_LEN = q_len, device = 'cuda',
                BLOCK_SIZE=128,_compile = False)

        return block_mask, ge_token_mask_1d_list

    def _prepare_decoder_attention_mask_t2i(self, attention_mask,ue_token_mask, ge_token_mask,scale_schedule, input_shape, inputs_embeds, past_key_values_length,image_gen_decoding):
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        # if input_shape[-1] > 1:
        combined_attention_mask,multi_scale_ge_lora_mask = _make_causal_mask_t2i(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
            ue_token_mask=ue_token_mask,
            ge_token_mask=ge_token_mask,
            scale_schedule  = scale_schedule ,
            image_gen_decoding = image_gen_decoding,
        )
        
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask,multi_scale_ge_lora_mask

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def _forward_t2i(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            image_gen_decoding: Optional[bool] = None,
            ue_token_mask: Optional[torch.Tensor] = None,
            ge_token_mask: Optional[torch.Tensor] = None,
            scale_schedule : Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)#.view(-1, seq_length)


        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)



        if  self.config.use_flex_for_t2i and self.training:
            if attention_mask is not None:
                attention_mask ,multi_scale_ge_lora_mask= self._prepare_decoder_flex_attention_mask_t2i(inputs_embeds,attention_mask,ue_token_mask,ge_token_mask,scale_schedule,image_gen_decoding)
        else:         
            attention_mask,multi_scale_ge_lora_mask = self._prepare_decoder_attention_mask_t2i(
                attention_mask, ue_token_mask, ge_token_mask,scale_schedule ,(batch_size, seq_length), inputs_embeds, past_key_values_length,image_gen_decoding)

        hidden_states = inputs_embeds
        multi_scale_ge_lora_mask_list = []
        if scale_schedule is not None:
            for j in range(len(scale_schedule)):
                multi_scale_ge_lora_mask_list.append(multi_scale_ge_lora_mask==scale_schedule[j][-1]*scale_schedule[j][-2]) # 10 B L
            for j in range(13-len(scale_schedule)): # the max number of scales : 13
                multi_scale_ge_lora_mask_list.append(multi_scale_ge_lora_mask==100000)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    image_gen_decoding,
                    ue_token_mask,
                    ge_token_mask,
                    multi_scale_ge_lora_mask_list
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    image_gen_decoding=image_gen_decoding,
                    ue_token_mask=ue_token_mask,
                    ge_token_mask=ge_token_mask,
                    multi_scale_ge_lora_mask=multi_scale_ge_lora_mask_list,

                )
            
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2VEForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image_gen_decoding : Optional[bool] = None,
        ue_token_mask: Optional[torch.Tensor] = None,
        ge_token_mask: Optional[torch.Tensor] = None,
        scale_schedule: Optional[torch.Tensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_gen_decoding = image_gen_decoding,
            ue_token_mask=ue_token_mask,
            ge_token_mask=ge_token_mask,
            scale_schedule=scale_schedule
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                'ue_token_mask': kwargs.get('ue_token_mask'),
                'ge_token_mask': kwargs.get('ge_token_mask')

            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

class GenMLP(nn.Module):
    def __init__(self, hidden_dim: int, image_gen_dim: int ) -> None:
        super().__init__()
        self.image_gen_dim = image_gen_dim
        self.mlp = nn.Sequential(
            nn.Linear( image_gen_dim , hidden_dim),
            nn.GELU(),
            nn.Linear( hidden_dim , hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self._init_linear_layers()

    def _init_linear_layers(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            if isinstance(layer,nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x.view(-1, self.image_gen_dim))
        return x


class GenHead(nn.Module):
    def __init__(self, hidden_dim: int, image_gen_dim: int ) -> None:
        super().__init__()
        self.image_gen_dim = image_gen_dim
        self.mlp = nn.Sequential(
            nn.Linear(image_gen_dim , image_gen_dim),
            nn.GELU(),
            nn.Linear(image_gen_dim, hidden_dim)
        )
        self._init_linear_layers()

    def _init_linear_layers(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('relu'))
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            if isinstance(layer,nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x.view(-1, self.image_gen_dim))
        return x
    

class PatchConvLayer(nn.Module):
    def __init__(self, config: PatchEmbeddingConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed
        
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype      
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = patch_embeds
        position_embedding = self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)
        embeddings = embeddings + position_embedding.to(target_dtype)
        # print(embeddings.shape)
        return embeddings


class PatchEmbeddingLayer(PreTrainedModel):
    main_input_name = 'pixel_values'
    config_class = PatchEmbeddingConfig

    def __init__(self, config: PatchEmbeddingConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = PatchConvLayer(config)
    
    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        pos_emb = self.embeddings.position_embedding
        _, _, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :].reshape(1, old_size // patch_size, old_size // patch_size, -1).permute(0, 3, 1, 2)
        pos_emb = F.interpolate(pos_emb.float(), size=new_size // patch_size, mode='bicubic', align_corners=False)
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        self.embeddings.position_embedding = nn.Parameter(pos_emb)
        self.embeddings.image_size = new_size
        logger.info('Resized position embeddings from {} to {}'.format(old_size, new_size))

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError('You have to specify pixel_values')

        if len(pixel_values.shape) == 4:
            hidden_states = self.embeddings(pixel_values)
        else:
            raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')

        if not return_dict:
            return (hidden_states, None,None)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=None,
            hidden_states=None,
            attentions=None,
        )


class OneCatVLModel(PreTrainedModel):
    config_class = OneCatVLChatConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['PatchEmbeddingLayer', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']
    _supports_flash_attn_2 = True

    def __init__(self, config):
        super().__init__(config)
        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
    
        # Modules.
        config.llm_config._attn_implementation = 'eager' # TODO 
        self.patch_language_model = Qwen2VEForCausalLM(config.llm_config)
        self.patch_vision_model = PatchEmbeddingLayer(config.patch_vision_config)
        self.patch_size =  config.patch_vision_config.patch_size
        self.template = config.template
        self.downsample_ratio = config.downsample_ratio
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message

        # Visual MLP
        patch_hidden_size = config.patch_vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        self.mlp = nn.Sequential(
            nn.LayerNorm(patch_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(patch_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        # Generation MLP
        self.image_gen_projector = GenMLP(llm_hidden_size, config.vae_embed_dim)
        self.image_gen_out_projector = GenHead(config.vae_embed_dim*2, llm_hidden_size)
        self.sos_embedding = nn.Embedding(1, config.vae_embed_dim)
        self.lvl_embed = nn.Embedding(15, llm_hidden_size) # TODO

        # SSA
        self.add_ssa_for_gen_ffn()
        
    def add_ssa_for_gen_ffn(self, r=64, lora_alpha=128, lora_dropout=0.1,lora_num=13):
        target_modules = ['mlp_ge.gate_proj', 'mlp_ge.up_proj', 'mlp_ge.down_proj']
        modules_to_replace = []
        adapter_name = 'default'
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
                parent_name, module_name = name.rsplit('.', 1) if '.' in name else ('', name)
                modules_to_replace.append((parent_name, module_name, module))
        
        for parent_name, module_name, original_module in modules_to_replace:
            new_module = Linear(original_module, adapter_name, r, lora_alpha, lora_dropout,lora_num)            
            if parent_name:
                parent_module = self.get_submodule(parent_name)
                setattr(parent_module, module_name, new_module)
            else:
                setattr(self, module_name, new_module)

    def get_sos_word_embeddings(self, input_ids, B):
        label_image_gen_start  = torch.LongTensor([0]).to(input_ids.device)
        label_image_gen_start = label_image_gen_start.repeat(B)  
        sos_embedding  = self.sos_embedding(label_image_gen_start)
        return sos_embedding

    def _prepare_text_inputs_for_t2i(self, input_ids, attention_mask):
        """Prepares text embeddings and position IDs."""
        input_embeds = self.patch_language_model.get_input_embeddings()(input_ids).clone()
        position_ids, _ = self.get_rope_index(input_ids, None, attention_mask)
        return input_embeds, position_ids
    
    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_hw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 2D rope index based on image's height and width in LLM, and 1D rope index for text.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide it.
            image_grid_hw (`torch.LongTensor` of shape `(num_images, 2)`, *optional*):
                The height and width of feature shape of each image in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(2, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """

        image_token_id = self.img_gen_context_token_id
        mrope_position_deltas = []
        if image_grid_hw is not None:
            total_input_ids = input_ids
            position_ids = torch.ones(
                2, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
                
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i] == 1]
                image_nums = 0
                image_nums = image_grid_hw.shape[0]
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                scale_index = 0
                remain_images = image_nums
                max_h,max_w= image_grid_hw[-1][0],image_grid_hw[-1][1]
               
                for _ in range(image_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    h, w = image_grid_hw[scale_index][0], image_grid_hw[scale_index][1]
                    scale_index += 1
                    remain_images -= 1
                    ed = ed_image
                    llm_grid_h, llm_grid_w = h.item() , w.item() 
                    if scale_index ==1:
                        text_len = ed - st
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(2, -1) + st_idx)

                    h_index = (torch.arange(llm_grid_h)*(max_h/llm_grid_h)).view(1, -1, 1).expand(1, llm_grid_h, llm_grid_w).flatten().round().int()
                    w_index = (torch.arange(llm_grid_w)*(max_w/llm_grid_w)).view(1, 1, -1).expand(1, llm_grid_h, llm_grid_w).flatten().round().int()
                    llm_pos_ids_list.append(torch.stack([h_index, w_index]) + text_len)
                    st = ed + llm_grid_h * llm_grid_w
   
                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(2, -1) + st_idx)
     
                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(2, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
  
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(2, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(2, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas
        
    def add_scale_embedding(self, feature, scale_ind, scale_schedule, need_to_pad=0):
        bs, seq_len, c = feature.shape
        patch_t, patch_h, patch_w = scale_schedule[scale_ind]
        t_mul_h_mul_w = patch_t * patch_h * patch_w
        assert t_mul_h_mul_w + need_to_pad == seq_len
        feature[:, :t_mul_h_mul_w] += self.lvl_embed(scale_ind*torch.ones((bs, t_mul_h_mul_w),dtype=torch.int).to(feature.device))
        return feature
    
    def chat(self, tokenizer, pixel_values, question, generation_config, pixel_values_thumbnail=None, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question
        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic image batch size: {image_bs}')
        patch_size = self.patch_size 
        token_h=int((pixel_values.shape[-2]//patch_size)//2)
        token_w=int((pixel_values.shape[-1]//patch_size)//2)
            
        num_image_token = token_h*token_w
        if pixel_values_thumbnail is not None:
            num_image_token+=256
        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_thumbnail=pixel_values_thumbnail,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            pixel_values_thumbnail: Optional[torch.FloatTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if self.patch_language_model is not None:
                input_patch_embeds = self.patch_language_model.get_input_embeddings()(input_ids)
            B, N, C = input_patch_embeds.shape
            input_patch_embeds = input_patch_embeds.reshape(B * N, C)
            if visual_features is not None:
                image_patch_embeds = visual_features
            else:
                image_patch_embeds = [self.extract_patch_feature(pixel_values).reshape(-1, C)]
                if pixel_values_thumbnail is not None:
                    image_patch_embeds += [self.extract_patch_feature(pixel_values_thumbnail).reshape(-1, C)]
                image_patch_embeds = torch.cat(image_patch_embeds)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_patch_embeds[selected] = image_patch_embeds.reshape(-1, C).to(input_patch_embeds.device)

            input_patch_embeds = input_patch_embeds.reshape(B, N, C)

            ue_token_mask = selected.reshape(B,N,1).to(input_patch_embeds.dtype)       
        else:
            input_patch_embeds = self.patch_language_model.get_input_embeddings()(input_ids)
            B, N, C = input_patch_embeds.shape
            ue_token_mask = torch.zeros_like(input_ids).reshape(B, N, 1).to(input_patch_embeds.dtype)

        outputs = self.patch_language_model.generate(
            inputs_embeds=input_patch_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            ue_token_mask=ue_token_mask,
            **generate_kwargs,
        )

        return outputs

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.reshape(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.reshape(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))

        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_patch_feature(self, pixel_values):
        for_band=False # TODO
        if isinstance(pixel_values,list):
            for_band=True
            pixel_values = pixel_values[0]
            b,num_band, c,h,w = pixel_values.shape
            pixel_values = pixel_values.reshape(b*num_band,c,h,w)
        patch_size = self.patch_vision_model.embeddings.patch_size
        token_height = int(pixel_values.shape[-2]//patch_size)
        if token_height%2==1:
            token_height+=1
        token_width = int(pixel_values.shape[-1]//patch_size)
        if token_width%2==1:
            token_width+=1
        image_patch_embeds = self.patch_vision_model(
            pixel_values=pixel_values,
            return_dict=True).last_hidden_state

        image_patch_embeds = image_patch_embeds.reshape(image_patch_embeds.shape[0], token_height, token_width, -1)
        image_patch_embeds = self.pixel_shuffle(image_patch_embeds, scale_factor=self.downsample_ratio)
        image_patch_embeds = image_patch_embeds.reshape(image_patch_embeds.shape[0], -1, image_patch_embeds.shape[-1])
        if for_band:
            image_patch_embeds = image_patch_embeds.reshape(b,num_band,image_patch_embeds.shape[-2],image_patch_embeds.shape[-1])
            image_patch_embeds = image_patch_embeds.reshape(b,-1,image_patch_embeds.shape[-1])
        image_patch_embeds = self.mlp(image_patch_embeds)
        return image_patch_embeds
    
    def generate_t2i(
        self,
        input_ids: torch.LongTensor = None,
        input_ids_cfg: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_cfg: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cfg: float = 1.0,
        top_k: int = 2,
        top_p: float = 0.97,
        h_div_w: float = 1.0,
    ):
        input_embeds,position_ids = self._prepare_text_inputs_for_t2i(input_ids, attention_mask)
        input_embeds_cfg,position_ids_cfg = self._prepare_text_inputs_for_t2i(input_ids_cfg, attention_mask_cfg)

        outputs = self.patch_language_model(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    ue_token_mask=None,
                    ge_token_mask=None,
                    image_gen_decoding=False
                )

        outputs_cfg = self.patch_language_model(
                    inputs_embeds=input_embeds_cfg,
                    attention_mask=attention_mask_cfg,
                    position_ids=position_ids_cfg,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    ue_token_mask=None,
                    ge_token_mask=None,
                    image_gen_decoding=False
                )
        kv_cache = outputs.past_key_values
        kv_cache_cfg = outputs_cfg.past_key_values
        B = input_ids.shape[0]
        tau = 0.5
        seed =int(time.time() * 1000)

        assert B==1, "batch size must be 1 for inference"
        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][self.vargpt_gen_args.pn]['scales']
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
        sos_embedding  = self.get_sos_word_embeddings(input_ids, B).unsqueeze(1)

        cfg_list = [cfg] * len(scale_schedule)
        tau_list = [tau] * len(scale_schedule)

        last_stage = sos_embedding
        vae_scale_schedule = scale_schedule

        summed_codes = 0

        text_len = position_ids.shape[-1]
        text_len_cfg = position_ids_cfg.shape[-1]

        max_h , max_w = scale_schedule[-1][-2] , scale_schedule[-1][-1] 
        rng = torch.Generator(device=position_ids.device)
        rng.manual_seed(seed)

        for stage_idx, pn in enumerate(scale_schedule):   # stage_idx: i-th scale
            llm_grid_h , llm_grid_w =  scale_schedule[stage_idx][-2] , scale_schedule[stage_idx][-1] 
            h_index = (torch.arange(llm_grid_h)*(max_h/llm_grid_h)).view(1, -1, 1).expand(1, llm_grid_h, llm_grid_w).flatten().round().int()
            w_index = (torch.arange(llm_grid_w)*(max_w/llm_grid_w)).view(1, 1, -1).expand(1, llm_grid_h, llm_grid_w).flatten().round().int()
            llm_pos_ids_list = (torch.stack([h_index, w_index]) + text_len).to(position_ids.device)
            llm_pos_ids_cfg_list = (torch.stack([h_index, w_index]) + text_len_cfg).to(position_ids.device)
            position_ids = llm_pos_ids_list.unsqueeze(1)
            position_ids_cfg = llm_pos_ids_cfg_list.unsqueeze(1)
            cfg = cfg_list[stage_idx]
    

            last_stage = self.image_gen_projector(last_stage).unsqueeze(0)
            x = self.add_scale_embedding(last_stage, stage_idx, scale_schedule)
                    
            outputs = self.patch_language_model(
                inputs_embeds=x,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=kv_cache,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                scale_schedule = scale_schedule,
                return_dict=return_dict,
                image_gen_decoding=True,
            )

            outputs_cfg = self.patch_language_model(
                inputs_embeds=x,
                attention_mask=None,
                position_ids=position_ids_cfg,
                past_key_values=kv_cache_cfg,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                scale_schedule = scale_schedule,
                return_dict=return_dict,
                image_gen_decoding=True,
            )
            kv_cache = outputs.past_key_values
            kv_cache_cfg = outputs_cfg.past_key_values
            hidden_states = outputs.hidden_states[-1]
            hidden_states_cfg = outputs_cfg.hidden_states[-1]

            logits_gen = self.image_gen_out_projector(hidden_states).unsqueeze(0).mul(1/tau_list[stage_idx])
            logits_gen_cfg = self.image_gen_out_projector(hidden_states_cfg).unsqueeze(0).mul(1/tau_list[stage_idx])

            logits_BlV = cfg * logits_gen + (1-cfg) * logits_gen_cfg

            tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
            logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
            idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
        
            idx_Bld_list = []

            assert pn[0] == 1
            idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1) # shape: [B, h, w, d] 
            
            idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] 

            idx_Bld_list.append(idx_Bld)
            codes = self.vae_local.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w]
            if stage_idx != len(scale_schedule)-1:
                summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=self.vae_local.quantizer.z_interplote_up)
                last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[stage_idx+1], mode=self.vae_local.quantizer.z_interplote_up) # [B, d, 1, h, w]
                last_stage = last_stage.squeeze(-3) # [B, d, h, w] 
            
                last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] 
                last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] 
                last_stage =  last_stage.to(dtype= hidden_states.dtype)
            else:
                summed_codes += codes

        recon_B3HW = self.vae_local.decode(summed_codes.squeeze(-3).to(dtype=last_stage.dtype)).to(dtype=torch.float32) 
        return recon_B3HW # BCHW, [-1,1]
    
    def generate_edit(
        self,
        pixel_values: torch.FloatTensor= None,
        input_ids: torch.LongTensor = None,
        input_ids_cfg: torch.LongTensor = None,
        input_ids_cfg2: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_cfg: Optional[torch.Tensor] = None,
        attention_mask_cfg2: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cfg_I: float = 1.0,
        cfg_T: float = 1.0,
        top_k: int = 2,
        top_p: float = 0.97,
        h_div_w: float = 1.0
    ):

        input_embeds,position_ids = self._prepare_text_inputs_for_t2i(input_ids, attention_mask)
        B,N,C = input_embeds.shape   
        assert B==1, "batch size must be 1 for inference"

        input_embeds_cfg,position_ids_cfg = self._prepare_text_inputs_for_t2i(input_ids_cfg, attention_mask_cfg)
        input_embeds_cfg2,position_ids_cfg2 = self._prepare_text_inputs_for_t2i(input_ids_cfg2, attention_mask_cfg2)
        

        image_patch_embeds = self.extract_patch_feature(pixel_values).reshape(-1,C)

        ref_img_context_selected = (input_ids == self.ref_img_context_token_id)
        ue_token_mask = ref_img_context_selected.reshape(B,N,1).to(input_embeds.dtype)
        input_embeds[ref_img_context_selected] = input_embeds[ref_img_context_selected] * 0.0 + image_patch_embeds.reshape(-1, C)
        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.patch_language_model(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    ue_token_mask=ue_token_mask,
                    ge_token_mask =None,
                    image_gen_decoding=False,
                )
        outputs_cfg = self.patch_language_model(
                    inputs_embeds=input_embeds_cfg,
                    attention_mask=attention_mask_cfg,
                    position_ids=position_ids_cfg,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    ue_token_mask=None,
                    ge_token_mask =None,
                    image_gen_decoding=False
                )
        outputs_cfg2 = self.patch_language_model(
                    inputs_embeds=input_embeds_cfg2,
                    attention_mask=attention_mask_cfg2,
                    position_ids=position_ids_cfg2,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    ue_token_mask=None,
                    ge_token_mask =None,
                    image_gen_decoding=False
                )

    
        tau = 0.5

        seed = int(time.time() * 1000)

        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][self.vargpt_gen_args.pn]['scales']
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
        g_seed = seed
        sos_embedding  = self.get_sos_word_embeddings(input_ids, B).unsqueeze(1)

        tau_list = [tau] * len(scale_schedule)

        last_stage = sos_embedding
        vae_scale_schedule = scale_schedule


        summed_codes = 0

        text_len = position_ids.shape[-1]
        text_len_cfg = position_ids_cfg.shape[-1]
        text_len_cfg2 = position_ids_cfg2.shape[-1]

        max_h , max_w = scale_schedule[-1][-2] , scale_schedule[-1][-1] 
        rng = torch.Generator(device=position_ids.device)
        rng.manual_seed(g_seed)

        for stage_idx, pn in enumerate(scale_schedule):   # stage_idx: i-th segment
            llm_grid_h , llm_grid_w =  scale_schedule[stage_idx][-2] , scale_schedule[stage_idx][-1] 
            h_index = (torch.arange(llm_grid_h)*(max_h/llm_grid_h)).view(1, -1, 1).expand(1, llm_grid_h, llm_grid_w).flatten().round().int()
            w_index = (torch.arange(llm_grid_w)*(max_w/llm_grid_w)).view(1, 1, -1).expand(1, llm_grid_h, llm_grid_w).flatten().round().int()

            llm_pos_ids_list = (torch.stack([h_index, w_index]) + text_len).to(position_ids.device)
            position_ids = llm_pos_ids_list.unsqueeze(1)

            llm_pos_ids_cfg_list = (torch.stack([h_index, w_index]) + text_len_cfg).to(position_ids.device)
            position_ids_cfg = llm_pos_ids_cfg_list.unsqueeze(1)

            llm_pos_ids_cfg_list2 = (torch.stack([h_index, w_index]) + text_len_cfg2).to(position_ids.device)
            position_ids_cfg2 = llm_pos_ids_cfg_list2.unsqueeze(1)

            last_stage = self.image_gen_projector(last_stage).unsqueeze(0)
            x = self.add_scale_embedding(last_stage, stage_idx, scale_schedule)
                             
           
            outputs = self.patch_language_model(
                inputs_embeds=x,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=outputs.past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                scale_schedule = scale_schedule,
                return_dict=return_dict,
                image_gen_decoding=True,
            )
            outputs_cfg = self.patch_language_model(
                inputs_embeds=x,
                attention_mask=None,
                position_ids=position_ids_cfg,
                past_key_values=outputs_cfg.past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                scale_schedule = scale_schedule,
                return_dict=return_dict,
                image_gen_decoding=True,
            )
            outputs_cfg2 = self.patch_language_model(
                inputs_embeds=x,
                attention_mask=None,
                position_ids=position_ids_cfg2,
                past_key_values=outputs_cfg2.past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                scale_schedule = scale_schedule,
                return_dict=return_dict,
                image_gen_decoding=True,
            ) 
         

            hidden_states = outputs.hidden_states[-1]
            logits_gen = self.image_gen_out_projector(hidden_states).unsqueeze(0).mul(1/tau_list[stage_idx])
           
            hidden_states_cfg = outputs_cfg.hidden_states[-1]
            logits_gen_cfg = self.image_gen_out_projector(hidden_states_cfg).unsqueeze(0).mul(1/tau_list[stage_idx])

            hidden_states_cfg2 = outputs_cfg2.hidden_states[-1]
            logits_gen_cfg2 = self.image_gen_out_projector(hidden_states_cfg2).unsqueeze(0).mul(1/tau_list[stage_idx])


            logits_cfg_I = (logits_gen+cfg_I*logits_gen_cfg)/(1+cfg_I)

            logits_BlV = logits_gen_cfg2 + cfg_T*(logits_cfg_I-logits_gen_cfg2)


            tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
            logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
            idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
        
            idx_Bld_list = []

            assert pn[0] == 1
            idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1) # shape: [B, h, w, d] 
            
            idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d]

            idx_Bld_list.append(idx_Bld)
            codes = self.vae_local.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] 
            if stage_idx != len(scale_schedule)-1:
                summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=self.vae_local.quantizer.z_interplote_up)
                last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[stage_idx+1], mode=self.vae_local.quantizer.z_interplote_up) # [B, d, 1, h, w] 
                last_stage = last_stage.squeeze(-3) # [B, d, h, w] 
            
                last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] 
                last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d]
                last_stage =  last_stage.to(dtype= hidden_states.dtype)
            else:
                summed_codes += codes     
            

        recon_B3HW =self.vae_local.decode(summed_codes.squeeze(-3).to(dtype=last_stage.dtype)).to(dtype=torch.float32) 
        return recon_B3HW # BCHW, [-1,1]

__all__ = ["OneCatVLModel"]
