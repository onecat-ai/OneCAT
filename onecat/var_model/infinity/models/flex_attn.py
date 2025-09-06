"""
Wrap torch's flex attention and handle mess info or potentially refactor
"""
from functools import partial
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    flex_attention_available = True
except ImportError:
    print(f"[Warning] flex attention need pytorch 2.5.0+ but your version is {torch.__version__}")
    flex_attention_available = False

def _causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def _length_to_offsets(lengths, device):
    """Converts a list of lengths to a list of offsets.

    Args:
        lengths: A list of lengths.

    """
    offsets = [0]
    offsets.extend(lengths)
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets

def _generate_var_mask_mod(offsets):
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        offsets: This tensor should be of shape(num_documents + 1)
            this should contain the cumulative counts of document tokens.
            e.g. if you have 3 documents of length 2, 4, 3 then
            offsets = [0, 2, 6, 9]

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.
    """

    def _offsets_to_doc_ids_tensor(offsets):
        device = offsets.device
        counts = offsets[1:] - offsets[:-1]
        return torch.repeat_interleave(
            torch.arange(len(counts), device=device, dtype=torch.int32), counts
        )

    document_id = _offsets_to_doc_ids_tensor(offsets)

    def var_mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_id[q_idx] == document_id[kv_idx]
        causal_mask = _causal_mask(b, h, q_idx, kv_idx)
        return same_doc | causal_mask

    return var_mask_mod

def _generate_var_infer_mask_with_kv_cache(lengths):
    kv_len = sum(lengths)
    def var_mask_mod(b, h, q_idx, kv_idx):
        return kv_idx < kv_len

    return var_mask_mod

class FlexAttn(nn.Module):
    def __init__(
            self, block_scales:list, mask_type:str, B, H, L:int, auto_padding=False
    ):
        """
        :param block_scales: accept VAR's block sizes like [(1,1), (2,2), (3,3)]
        :param mask_type: var/causal
        :param B: batch size
        :param H: heads num
        :param L: sequence length
        """
        super().__init__()
        if not flex_attention_available:
            raise NotImplementedError((f"[Error] flex attention need pytorch 2.5.0+ but your version is {torch.__version__}"))

        self.support_mask_type = ["var", "causal", "var_infer_mask_with_kv_cache"]
        self.auto_padding = auto_padding

        self.flex_attention = torch.compile(flex_attention)

        self.block_scales = block_scales
        self.lengths = [ x * y * z for x,y,z in block_scales]

        self.offsets = _length_to_offsets(self.lengths, device='cuda')

        # if L paded to align 128, block need to cover padding area
        if self.offsets[-1] < L:
            self.offsets = torch.cat((self.offsets, torch.tensor([L], device='cuda')), dim=0)

        if mask_type == "var":
            self.mask_mod = _generate_var_mask_mod(self.offsets)
            self.block_mask = create_block_mask(self.mask_mod, B = B, H = H, Q_LEN = L, KV_LEN = L, device = 'cuda', _compile = True)
        elif mask_type == "causal":
            self.mask_mod = _causal_mask
            self.block_mask = create_block_mask(self.mask_mod, B = B, H = H, Q_LEN = L, KV_LEN = L, device = 'cuda', _compile = True)
        elif mask_type == 'var_infer_mask_with_kv_cache':
            self.mask_mod = _generate_var_infer_mask_with_kv_cache(self.lengths)
            self.block_mask = create_block_mask(self.mask_mod, B = B, H = H, Q_LEN = L, KV_LEN = L, device = 'cuda', _compile = True)
        else:
            raise NotImplementedError(f"{mask_type} not supportted in FlexAttn, support type:{self.support_mask_type}")


    def forward(self, q, k, v, scale = None):
        if self.auto_padding:
            q_pad_len = (128 - q.shape[-2] % 128) % 128
            kv_pad_len = (128 - k.shape[-2] % 128) % 128
            q_pad = F.pad(q, (0, 0, 0, q_pad_len))
            k_pad = F.pad(k, (0, 0, 0, kv_pad_len))
            v_pad = F.pad(v, (0, 0, 0, kv_pad_len))
            oup = self.flex_attention(q_pad.to(v_pad.dtype), k_pad.to(v.dtype), v_pad, block_mask = self.block_mask, scale = scale)
            if q_pad_len > 0:
                oup = oup[:,:,:-q_pad_len]
        else:
            oup = self.flex_attention(q.to(v.dtype), k.to(v.dtype), v, block_mask = self.block_mask, scale = scale)
        return oup

    def extra_repr(self) -> str:
        tail = ''
        return f'block size:{self.block_scales} {tail}'
