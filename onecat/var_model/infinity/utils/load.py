#!/usr/bin/python3
import gc
import os
import os.path as osp
import random
import sys
from copy import deepcopy
from typing import Tuple, Union

import colorama
import torch
import yaml

import infinity.utils.dist as dist

from vargpt_qwen_v1_1.var_model.infinity.models import Infinity
from vargpt_qwen_v1_1.var_model.infinity.models.ema import get_ema_model
from vargpt_qwen_v1_1.var_model.infinity.utils import arg_util, misc
from vargpt_qwen_v1_1.var_model.infinity.utils.misc import os_system


def build_vae_gpt(args: arg_util.Args, vae_st: dict, skip_gpt: bool, force_flash=False, device='cuda'):
    if args.vae_type in [8,16,18,20,24,32,64,128]:
        from vargpt_qwen_v1_1.var_model.infinity.models.bsq_vae.vae import vae_model
        schedule_mode = "dynamic"
        codebook_dim = args.vae_type # 18
        codebook_size = 2**codebook_dim
        if args.apply_spatial_patchify:
            patch_size = 8
            encoder_ch_mult=[1, 2, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4]
        else:
            patch_size = 16
            encoder_ch_mult=[1, 2, 4, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4, 4]
        vae_local = vae_model(vae_st, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size, 
                              encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(args.device)
        if args.fake_vae_input:
            vae_local.encoder = None
            vae_local.decoder = None
            torch.cuda.empty_cache()
    else:
        raise ValueError(f"vae_type {args.vae_type} not supported")
    if force_flash: args.flash = True
    gpt_kw = dict(
        pretrained=False, global_pool='',
        text_channels=args.Ct5, text_maxlen=args.tlen,
        norm_eps=args.norm_eps, rms_norm=args.rms,
        shared_aln=args.saln, head_aln=args.haln,
        cond_drop_rate=args.cfg, rand_uncond=args.rand_uncond, drop_rate=args.drop,
        cross_attn_layer_scale=args.ca_gamma, nm0=args.nm0, tau=args.tau, cos_attn=args.cos, swiglu=args.swi,
        raw_scale_schedule=args.scale_schedule,
        head_depth=args.dec,
        top_p=args.tp, top_k=args.tk,
        customized_flash_attn=args.flash, fused_mlp=args.fuse, fused_norm=args.fused_norm,
        checkpointing=args.enable_checkpointing,
        pad_to_multiplier=args.pad_to_multiplier,
        use_flex_attn=args.use_flex_attn,
        batch_size=args.batch_size,
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block,
        use_bit_label=args.use_bit_label,
        rope2d_each_sa_layer=args.rope2d_each_sa_layer,
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        pn=args.pn,
        train_h_div_w_list=args.train_h_div_w_list,
        always_training_scales=args.always_training_scales,
        apply_spatial_patchify=args.apply_spatial_patchify,
    )
    if args.dp >= 0: gpt_kw['drop_path_rate'] = args.dp
    if args.hd > 0: gpt_kw['num_heads'] = args.hd
    
    print(f'[create gpt_wo_ddp] constructor kw={gpt_kw}\n')
    gpt_kw['vae_local'] = vae_local
    
    model_str = args.model.replace('vgpt', 'infinity')   # legacy
    print(f"{model_str=}")
    if model_str.rsplit('c', maxsplit=1)[-1].isdecimal():
        model_str, block_chunks = model_str.rsplit('c', maxsplit=1)
        block_chunks = int(block_chunks)
    else:
        block_chunks = 1
    gpt_kw['block_chunks'] = block_chunks
    
    from vargpt_qwen_v1_1.var_model.infinity.models import Infinity
    from timm.models import create_model
    gpt_wo_ddp: Infinity = create_model(model_str, **gpt_kw)
    if args.use_fsdp_model_ema:
        gpt_wo_ddp_ema = get_ema_model(gpt_wo_ddp)
    else:
        gpt_wo_ddp_ema = None
    gpt_wo_ddp = gpt_wo_ddp.to(device)

    assert all(not p.requires_grad for p in vae_local.parameters())
    assert all(p.requires_grad for n, p in gpt_wo_ddp.named_parameters())
    
    return vae_local, gpt_wo_ddp, gpt_wo_ddp_ema


if __name__ == '__main__':
    ld(sys.argv[1])
