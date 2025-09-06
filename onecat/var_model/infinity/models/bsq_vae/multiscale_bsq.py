"""
Binary Spherical Quantization
Proposed in https://arxiv.org/abs/2406.07548

In the simplest setup, each dimension is quantized into {-1, 1}.
An entropy penalty is used to encourage utilization.
"""

import random
from math import log2, ceil
from functools import partial, cache
from collections import namedtuple
from contextlib import nullcontext

import torch.distributed as dist
from torch.distributed import nn as dist_nn

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn import Module
from torch.amp import autocast
import numpy as np

from einops import rearrange, reduce, pack, unpack

# from einx import get_at

from .dynamic_resolution import predefined_HW_Scales_dynamic

# constants

Return = namedtuple('Return', ['quantized', 'indices', 'bit_indices', 'entropy_aux_loss'])

LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'batch_entropy', 'commitment'])

# distributed helpers

@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1

def maybe_distributed_mean(t):
    if not is_distributed():
        return t

    dist_nn.all_reduce(t)
    t = t / dist.get_world_size()
    return t

# helper functions

def exists(v):
    return v is not None

def identity(t):
    return t

def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None

def round_up_multiple(num, mult):
    return ceil(num / mult) * mult

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def l2norm(t):
    return F.normalize(t, dim = -1)

# entropy

def log(t, eps = 1e-5):
    return t.clamp(min = eps).log()

def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)

# cosine sim linear

class CosineSimLinear(Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        scale = 1.
    ):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(dim_in, dim_out))

    def forward(self, x):
        x = F.normalize(x, dim = -1)
        w = F.normalize(self.weight, dim = 0)
        return (x @ w) * self.scale


def get_latent2scale_schedule(T: int, H: int, W: int, mode="original"):
    assert mode in ["original", "dynamic", "dense", "same1", "same2", "same3"]
    predefined_HW_Scales = {
        # 256 * 256
        (32, 32): [(1, 1), (2, 2), (3, 3), (4, 4), (6, 6), (9, 9), (13, 13), (18, 18), (24, 24), (32, 32)],
        (16, 16): [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (8, 8), (10, 10), (13, 13), (16, 16)],
        # 1024x1024
        (64, 64): [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (7, 7), (9, 9), (12, 12), (16, 16), (21, 21), (27, 27), (36, 36), (48, 48), (64, 64)],

        (36, 64): [(1, 1), (2, 2), (3, 3), (4, 4), (6, 6), (9, 12), (13, 16), (18, 24), (24, 32), (32, 48), (36, 64)],
    }
    if mode == "dynamic":
        predefined_HW_Scales.update(predefined_HW_Scales_dynamic)
    elif mode == "dense":
        predefined_HW_Scales[(16, 16)] = [(x, x) for x in range(1, 16+1)]
        predefined_HW_Scales[(32, 32)] = predefined_HW_Scales[(16, 16)] + [(20, 20), (24, 24), (28, 28), (32, 32)]
        predefined_HW_Scales[(64, 64)] = predefined_HW_Scales[(32, 32)] + [(40, 40), (48, 48), (56, 56), (64, 64)]
    elif mode.startswith("same"):
        num_quant = int(mode[len("same"):])
        predefined_HW_Scales[(16, 16)] = [(16, 16) for _ in range(num_quant)]
        predefined_HW_Scales[(32, 32)] = [(32, 32) for _ in range(num_quant)]
        predefined_HW_Scales[(64, 64)] = [(64, 64) for _ in range(num_quant)]

    predefined_T_Scales = [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17, 17, 17, 17, 17]
    patch_THW_shape_per_scale = predefined_HW_Scales[(H, W)]
    if len(predefined_T_Scales) < len(patch_THW_shape_per_scale):
        # print("warning: the length of predefined_T_Scales is less than the length of patch_THW_shape_per_scale!")
        predefined_T_Scales += [predefined_T_Scales[-1]] * (len(patch_THW_shape_per_scale) - len(predefined_T_Scales))
    patch_THW_shape_per_scale = [(min(T, t), h, w ) for (h, w), t in zip(patch_THW_shape_per_scale, predefined_T_Scales[:len(patch_THW_shape_per_scale)])]
    return patch_THW_shape_per_scale

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    normalized_shape: int
    """
    def __init__(self, normalized_shape, norm_weight=False, eps=1e-6, data_format="channels_first"):
        super().__init__()
        if norm_weight:
            self.weight = nn.Parameter(torch.ones(normalized_shape)/(normalized_shape**0.5))
        else:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            if x.ndim == 4: # (b, c, h, w)
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            elif x.ndim == 5: # (b, c, t, h, w)
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            else:
                raise ValueError("the number of dimensions of the input should be 4 or 5")
            return x

class MultiScaleBSQ(Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """

    def __init__(
        self,
        *,
        dim,
        codebook_size,
        soft_clamp_input_value = None,
        aux_loss = False, # intermediate auxiliary loss
        ln_before_quant=False, # add a LN before multi-scale RQ
        ln_init_by_sqrt=False, # weight init by 1/sqrt(d)
        use_decay_factor=False,
        use_stochastic_depth=False,
        drop_rate=0.,
        schedule_mode="original", # ["original", "dynamic", "dense"]
        keep_first_quant=False,
        keep_last_quant=False,
        remove_residual_detach=False,
        random_flip = False,
        flip_prob = 0.5,
        flip_mode = "stochastic", # "stochastic", "deterministic"
        max_flip_lvl = 1,
        random_flip_1lvl = False, # random flip one level each time
        flip_lvl_idx = None,
        drop_when_test=False,
        drop_lvl_idx=None,
        drop_lvl_num=0,
        **kwargs
    ):
        super().__init__()
        codebook_dim = int(log2(codebook_size))

        requires_projection = codebook_dim != dim
        self.project_in = nn.Linear(dim, codebook_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_dim, dim) if requires_projection else nn.Identity()
        self.has_projections = requires_projection
        self.layernorm = LayerNorm(codebook_dim, norm_weight=ln_init_by_sqrt) if ln_before_quant else nn.Identity()
        self.use_stochastic_depth = use_stochastic_depth
        self.drop_rate = drop_rate
        self.remove_residual_detach = remove_residual_detach
        self.random_flip = random_flip
        self.flip_prob = flip_prob
        self.flip_mode = flip_mode
        self.max_flip_lvl = max_flip_lvl
        self.random_flip_1lvl = random_flip_1lvl
        self.flip_lvl_idx = flip_lvl_idx
        assert (random_flip and random_flip_1lvl) == False
        self.drop_when_test = drop_when_test
        self.drop_lvl_idx = drop_lvl_idx
        self.drop_lvl_num = drop_lvl_num
        if self.drop_when_test:
            assert drop_lvl_idx is not None
            assert drop_lvl_num > 0

        self.lfq = BSQ(
            dim = codebook_dim,
            codebook_scale = 1/np.sqrt(codebook_dim),
            soft_clamp_input_value = soft_clamp_input_value,
            # experimental_softplus_entropy_loss=True,
            # entropy_loss_offset=2,
            **kwargs
        )

        self.z_interplote_up = 'trilinear'
        self.z_interplote_down = 'area'
        
        self.use_decay_factor = use_decay_factor
        self.schedule_mode = schedule_mode
        self.keep_first_quant = keep_first_quant
        self.keep_last_quant = keep_last_quant
        if self.use_stochastic_depth and self.drop_rate > 0:
            assert self.keep_first_quant or self.keep_last_quant

    @property
    def codebooks(self):
        return self.lfq.codebook

    def get_codes_from_indices(self, indices_list):
        all_codes = []
        for indices in indices_list:
            codes = self.lfq.indices_to_codes(indices)
            all_codes.append(codes)
        _, _, T, H, W = all_codes[-1].size()
        summed_codes = 0
        for code in all_codes:
            summed_codes += F.interpolate(code, size=(T, H, W), mode=self.z_interplote_up)
        return summed_codes

    def get_output_from_indices(self, indices):
        codes = self.get_codes_from_indices(indices)
        codes_summed = reduce(codes, 'q ... -> ...', 'sum')
        return self.project_out(codes_summed)

    def flip_quant(self, x):
        assert self.flip_mode == 'stochastic'
        flip_mask = torch.rand_like(x) < self.flip_prob
        x = x.clone()
        x[flip_mask] = -x[flip_mask]
        return x

    def forward(
        self,
        x,
        scale_schedule=None,
        mask = None,
        return_all_codes = False,
        return_residual_norm_per_scale = False
    ):
        if x.ndim == 4:
            x = x.unsqueeze(2)
        B, C, T, H, W = x.size()    

        if scale_schedule is None:
            if self.schedule_mode.startswith("same"):
                scale_num = int(self.schedule_mode[len("same"):])
                assert T == 1
                scale_schedule = [(1, H, W)] * scale_num
            else:
                scale_schedule = get_latent2scale_schedule(T, H, W, mode=self.schedule_mode)
                scale_num = len(scale_schedule)

        # x = self.project_in(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous() # (b, c, t, h, w) => (b, t, h, w, c)
        x = self.project_in(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous() # (b, t, h, w, c) => (b, c, t, h, w) 
        x = self.layernorm(x)

        quantized_out = 0.
        residual = x

        all_losses = []
        all_indices = []
        all_bit_indices = []
        var_inputs = []
        residual_norm_per_scale = []
        
        # go through the layers
        out_fact = init_out_fact = 1.0
        # residual_list = []
        # interpolate_residual_list = []
        # quantized_list = []
        if self.drop_when_test:
            drop_lvl_start = self.drop_lvl_idx
            drop_lvl_end = self.drop_lvl_idx + self.drop_lvl_num
        scale_num = len(scale_schedule)
        with autocast('cuda', enabled = False):
            for si, (pt, ph, pw) in enumerate(scale_schedule):
                out_fact = max(0.1, out_fact) if self.use_decay_factor else init_out_fact
                if (pt, ph, pw) != (T, H, W):
                    interpolate_residual = F.interpolate(residual, size=(pt, ph, pw), mode=self.z_interplote_down)
                else:
                    interpolate_residual = residual
                if return_residual_norm_per_scale:
                    residual_norm_per_scale.append((torch.abs(interpolate_residual) < 0.05 * self.lfq.codebook_scale).sum() / interpolate_residual.numel())
                # residual_list.append(torch.norm(residual.detach(), dim=1).mean())
                # interpolate_residual_list.append(torch.norm(interpolate_residual.detach(), dim=1).mean())
                if self.training and self.use_stochastic_depth and random.random() < self.drop_rate:
                    if (si == 0 and self.keep_first_quant) or (si == scale_num - 1 and self.keep_last_quant):
                        quantized, indices, _, loss = self.lfq(interpolate_residual)
                        quantized = quantized * out_fact
                        all_indices.append(indices)
                        all_losses.append(loss)
                    else:
                        quantized = torch.zeros_like(interpolate_residual)
                elif self.drop_when_test and drop_lvl_start <= si < drop_lvl_end:
                    continue                     
                else:
                    # residual_norm = torch.norm(interpolate_residual.detach(), dim=1) # (b, t, h, w)
                    # print(si, residual_norm.min(), residual_norm.max(), residual_norm.mean())
                    quantized, indices, bit_indices, loss = self.lfq(interpolate_residual)
                    if self.random_flip and si < self.max_flip_lvl:
                        quantized = self.flip_quant(quantized)
                    if self.random_flip_1lvl and si == self.flip_lvl_idx:
                        quantized = self.flip_quant(quantized)
                    quantized = quantized * out_fact
                    all_indices.append(indices)
                # quantized_list.append(torch.norm(quantized.detach(), dim=1).mean())
                if (pt, ph, pw) != (T, H, W):
                    quantized = F.interpolate(quantized, size=(T, H, W), mode=self.z_interplote_up).contiguous()
                
                if self.remove_residual_detach:
                    residual = residual - quantized
                else:
                    residual = residual - quantized.detach()
                quantized_out = quantized_out + quantized

                all_bit_indices.append(bit_indices)
                all_losses.append(loss)
                if si != scale_num - 1:
                    var_inputs.append(F.interpolate(quantized_out, size=scale_schedule[si+1], mode=self.z_interplote_down).contiguous())
                
                if self.use_decay_factor:
                    out_fact -= 0.1
        # print("residual_list:", residual_list)
        # print("interpolate_residual_list:", interpolate_residual_list)
        # print("quantized_list:", quantized_list)
        # import ipdb; ipdb.set_trace()
        # project out, if needed
        quantized_out = quantized_out.permute(0, 2, 3, 4, 1).contiguous() # (b, c, t, h, w) => (b, t, h, w, c)
        quantized_out = self.project_out(quantized_out)
        quantized_out = quantized_out.permute(0, 4, 1, 2, 3).contiguous() # (b, t, h, w, c) => (b, c, t, h, w)

        # image
        if quantized_out.size(2) == 1:
            quantized_out = quantized_out.squeeze(2)

        # stack all losses and indices

        all_losses = torch.stack(all_losses, dim = -1)

        ret = (quantized_out, all_indices, all_bit_indices, residual_norm_per_scale, all_losses, var_inputs)

        if not return_all_codes:
            return ret

        # whether to return all codes from all codebooks across layers
        all_codes = self.get_codes_from_indices(all_indices)

        # will return all codes in shape (quantizer, batch, sequence length, codebook dimension)

        return (*ret, all_codes)


class BSQ(Module):
    def __init__(
        self,
        *,
        dim = None,
        codebook_size = None,
        entropy_loss_weight = 0.1,
        commitment_loss_weight = 0.25,
        diversity_gamma = 1.,
        straight_through_activation = nn.Identity(),
        num_codebooks = 1,
        keep_num_codebooks_dim = None,
        codebook_scale = 1.,                        # for residual LFQ, codebook scaled down by 2x at each layer
        frac_per_sample_entropy = 1.,               # make less than 1. to only use a random fraction of the probs for per sample entropy
        has_projections = None,
        projection_has_bias = True,
        soft_clamp_input_value = None,
        cosine_sim_project_in = False,
        cosine_sim_project_in_scale = None,
        channel_first = None,
        experimental_softplus_entropy_loss = False,
        entropy_loss_offset = 5.,                   # how much to shift the loss before softplus
        spherical = True,                          # from https://arxiv.org/abs/2406.07548
        force_quantization_f32 = True,               # will force the quantization step to be full precision
        inv_temperature = 100.0,
        gamma0=1.0, gamma=1.0, zeta=1.0,
        preserve_norm = False, # whether to preserve the original norm info
        new_quant = False, # new quant function，
        mask_out = False, # mask the output as 0 in some conditions
        use_out_phi = False, # use output phi network
        use_out_phi_res = False, # residual out phi
    ):
        super().__init__()

        # some assert validations

        assert exists(dim) or exists(codebook_size), 'either dim or codebook_size must be specified for LFQ'
        assert not exists(codebook_size) or log2(codebook_size).is_integer(), f'your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})'

        codebook_size = default(codebook_size, lambda: 2 ** dim)
        self.codebook_size = codebook_size

        codebook_dim = int(log2(codebook_size))
        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)
        self.codebook_dims = codebook_dims

        has_projections = default(has_projections, dim != codebook_dims)

        if cosine_sim_project_in:
            cosine_sim_project_in = default(cosine_sim_project_in_scale, codebook_scale)
            project_in_klass = partial(CosineSimLinear, scale = cosine_sim_project_in)
        else:
            project_in_klass = partial(nn.Linear, bias = projection_has_bias)

        self.project_in = project_in_klass(dim, codebook_dims) if has_projections else nn.Identity() # nn.Identity()
        self.project_out = nn.Linear(codebook_dims, dim, bias = projection_has_bias) if has_projections else nn.Identity() # nn.Identity()
        self.has_projections = has_projections

        self.out_phi = nn.Linear(codebook_dims, codebook_dims) if use_out_phi else nn.Identity()
        self.use_out_phi_res = use_out_phi_res
        if self.use_out_phi_res:
            self.out_phi_scale = nn.Parameter(torch.zeros(codebook_dims), requires_grad=True) # init as zero

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # channel first

        self.channel_first = channel_first

        # straight through activation

        self.activation = straight_through_activation

        # For BSQ (binary spherical quantization)
        if not spherical:
            raise ValueError("For BSQ, spherical must be True.")
        self.persample_entropy_compute = 'analytical'
        self.inv_temperature = inv_temperature
        self.gamma0 = gamma0  # loss weight for entropy penalty
        self.gamma = gamma  # loss weight for entropy penalty
        self.zeta = zeta    # loss weight for entire entropy penalty
        self.preserve_norm = preserve_norm
        self.new_quant = new_quant
        self.mask_out = mask_out

        # entropy aux loss related weights

        assert 0 < frac_per_sample_entropy <= 1.
        self.frac_per_sample_entropy = frac_per_sample_entropy

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # codebook scale

        self.codebook_scale = codebook_scale

        # commitment loss

        self.commitment_loss_weight = commitment_loss_weight

        # whether to soft clamp the input value from -value to value

        self.soft_clamp_input_value = soft_clamp_input_value
        assert not exists(soft_clamp_input_value) or soft_clamp_input_value >= codebook_scale

        # whether to make the entropy loss positive through a softplus (experimental, please report if this worked or not in discussions)

        self.entropy_loss_offset = entropy_loss_offset
        self.experimental_softplus_entropy_loss = experimental_softplus_entropy_loss

        # for no auxiliary loss, during inference

        self.register_buffer('mask', 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

        # whether to force quantization step to be f32

        self.force_quantization_f32 = force_quantization_f32

        # codes

        # all_codes = torch.arange(codebook_size)
        # bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        # codebook = self.bits_to_codes(bits)

        # self.register_buffer('codebook', codebook.float(), persistent = False)

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    # @property
    # def dtype(self):
    #     return self.codebook.dtype

    def indices_to_codes(
        self,
        indices,
        label_type = 'int_label',
        project_out = True
    ):
        assert label_type in ['int_label', 'bit_label']
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        should_transpose = default(self.channel_first, is_img_or_video)

        if not self.keep_num_codebooks_dim:
            if label_type == 'int_label':
                indices = rearrange(indices, '... -> ... 1')
            else:
                indices = indices.unsqueeze(-2)

        # indices to codes, which are bits of either -1 or 1

        if label_type == 'int_label':
            assert indices[..., None].int().min() > 0
            bits = ((indices[..., None].int() & self.mask) != 0).float() # .to(self.dtype)
        else:
            bits = indices

        codes = self.bits_to_codes(bits)

        codes = l2norm(codes) # must normalize when using BSQ

        codes = rearrange(codes, '... c d -> ... (c d)')

        # whether to project codes out to original dimensions
        # if the input feature dimensions were not log2(codebook size)

        if project_out:
            codes = self.project_out(codes)

        # rearrange codes back to original shape

        if should_transpose:
            codes = rearrange(codes, 'b ... d -> b d ...')

        return codes

    def quantize(self, z):
        assert z.shape[-1] == self.codebook_dims, f"Expected {self.codebook_dims} dimensions, got {z.shape[-1]}"

        zhat = torch.where(z > 0, 
                           torch.tensor(1, dtype=z.dtype, device=z.device), 
                           torch.tensor(-1, dtype=z.dtype, device=z.device))
        return z + (zhat - z).detach()

    def quantize_new(self, z):
        assert z.shape[-1] == self.codebook_dims, f"Expected {self.codebook_dims} dimensions, got {z.shape[-1]}"

        zhat = torch.where(z > 0, 
                           torch.tensor(1, dtype=z.dtype, device=z.device), 
                           torch.tensor(-1, dtype=z.dtype, device=z.device))

        q_scale = 1. / (self.codebook_dims ** 0.5)
        zhat = q_scale * zhat # on unit sphere

        return z + (zhat - z).detach()

    def soft_entropy_loss(self, z):
        if self.persample_entropy_compute == 'analytical':
            # if self.l2_norm:
            p = torch.sigmoid(-4 * z / (self.codebook_dims ** 0.5) * self.inv_temperature)
            # else:
            #     p = torch.sigmoid(-4 * z * self.inv_temperature)
            prob = torch.stack([p, 1-p], dim=-1) # (b, h, w, 18, 2)
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean() # (b,h,w,18)->(b,h,w)->scalar
        else:
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()

        # macro average of the probability of each subgroup
        avg_prob = reduce(prob, '... g d ->g d', 'mean') # (18, 2)
        codebook_entropy = self.get_entropy(avg_prob, dim=-1, normalize=False)

        # the approximation of the entropy is the sum of the entropy of each subgroup
        return per_sample_entropy, codebook_entropy.sum(), avg_prob

    def get_entropy(self, count, dim=-1, eps=1e-4, normalize=True):
        if normalize: # False
            probs = (count + eps) / (count + eps).sum(dim=dim, keepdim =True)
        else: # True
            probs = count
        H = -(probs * torch.log(probs + 1e-8)).sum(dim=dim)
        return H

    def forward(
        self,
        x,
        return_loss_breakdown = False,
        mask = None,
        entropy_weight=0.1
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = x.ndim >= 4
        should_transpose = default(self.channel_first, is_img_or_video)

        # standardize image or video into (batch, seq, dimension)

        if should_transpose:
            x = rearrange(x, 'b d ... -> b ... d')
            x, ps = pack_one(x, 'b * d') # x.shape [b, hwt, c]

        assert x.shape[-1] == self.dim, f'expected dimension of {self.dim} but received {x.shape[-1]}'

        x = self.project_in(x)

        # split out number of codebooks

        x = rearrange(x, 'b n (c d) -> b n c d', c = self.num_codebooks)

        x = l2norm(x)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32

        quantization_context = partial(autocast, 'cuda', enabled = False) if force_f32 else nullcontext

        indices = None
        with quantization_context():

            if force_f32:
                orig_dtype = x.dtype
                x = x.float()
            
            # use straight-through gradients (optionally with custom activation fn) if training
            if self.new_quant:
                quantized = self.quantize_new(x)

            # calculate indices
            bit_indices = (quantized > 0).int()
            entropy_penalty = persample_entropy = cb_entropy = self.zero
            commit_loss = self.zero

            # input back to original dtype if needed

            if force_f32:
                x = x.type(orig_dtype)

        # merge back codebook dim
        x = quantized # rename quantized to x for output
        x = rearrange(x, 'b n c d -> b n (c d)')

        # project out to feature dimension if needed

        x = self.project_out(x)

        # reconstitute image or video dimensions

        if should_transpose:
            x = unpack_one(x, ps, 'b * d')
            x = rearrange(x, 'b ... d -> b d ...')

            bit_indices = unpack_one(bit_indices, ps, 'b * c d')

        # whether to remove single codebook dim

        if not self.keep_num_codebooks_dim:
            bit_indices = rearrange(bit_indices, '... 1 d -> ... d')

        # complete aux loss

        aux_loss = commit_loss * self.commitment_loss_weight + (self.zeta * entropy_penalty / self.inv_temperature)*entropy_weight
        # returns

        ret = Return(x, indices, bit_indices, aux_loss)

        if not return_loss_breakdown:
            return ret

        return ret, LossBreakdown(persample_entropy, cb_entropy, commit_loss)

