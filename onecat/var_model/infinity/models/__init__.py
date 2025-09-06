import torch
from timm.loss import SoftTargetCrossEntropy

from timm.models.layers import DropPath

from .infinity import Infinity, sample_with_top_k_top_p_also_inplace_modifying_logits_

def _ex_repr(self):
    return ', '.join(
        f'{k}=' + (f'{v:g}' if isinstance(v, float) else str(v))
        for k, v in vars(self).items()
        if not k.startswith('_') and k != 'training'
        and not isinstance(v, (torch.nn.Module, torch.Tensor))
    )
for clz in (torch.nn.CrossEntropyLoss, SoftTargetCrossEntropy):  # no longer __repr__ DropPath with drop_prob
    if hasattr(clz, 'extra_repr'):
        clz.extra_repr = _ex_repr
    else:
        clz.__repr__ = lambda self: f'{type(self).__name__}({_ex_repr(self)})'

DropPath.__repr__ = lambda self: f'{type(self).__name__}(...)'

alias_dict = {}
for d in range(6, 40+2, 2):
    alias_dict[f'd{d}'] = f'infinity_d{d}'
alias_dict_inv = {v: k for k, v in alias_dict.items()}
