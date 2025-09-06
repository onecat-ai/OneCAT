import os
import os.path as osp
import time
import gc
import json
import math
import random
import sys
import argparse
import copy
# import traceback
import collections
from collections import deque
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.distributed as tdist
import tqdm

from tools.run_infinity import *
from infinity.dataset.dataset_t2i_iterable import T2IIterableDataset
from infinity.models.bitwise_self_correction import BitwiseSelfCorrection

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--reweight_loss_by_scale', type=int, default=1, choices=[0,1])
    parser.add_argument('--vis_model_flop_param', type=int, default=0, choices=[0,1])
    parser.add_argument('--meta_folder', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--dataloader_workers', type=int, default=12)
    parser.add_argument('--noise_apply_layers', type=int, default=20)
    parser.add_argument('--noise_apply_requant', type=int, default=1, choices=[0,1])
    parser.add_argument('--noise_apply_strength', type=float, default=0.2)
    parser.add_argument('--debug_bsc', type=int, default=0, choices=[0,1])
    parser.add_argument('--log_freq', type=int, default=10)
    args = parser.parse_args()
    
    # load text encoder
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    # load vae
    vae = load_visual_tokenizer(args)
    # load infinity
    infinity = load_transformer(vae, args)

    bitwise_self_correction = BitwiseSelfCorrection(vae, args)
    
    device = torch.device('cuda')
    dataset = T2IIterableDataset(
        args=None, 
        meta_folder=args.meta_folder,
        data_load_reso=None, 
        max_caption_len=512, 
        short_prob=0.0, 
        load_vae_instead_of_image=False,
        buffersize=100,
        seed=0, 
        online_t5=True,
        pn=args.pn,
        batch_size=args.batch_size,
        num_replicas=1,
        rank=0,
        dataloader_workers=args.dataloader_workers,
    )
    dataloader = DataLoader(dataset, batch_size=None, num_workers=args.dataloader_workers)
    print(f'len(dataloader): {len(dataloader)}, len(dataset): {len(dataset)}, total_samples: {dataset.total_samples()}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t1 = time.time()
    dataloader.dataset.set_epoch(0)
    pbar = tqdm.tqdm(total=len(dataloader))
    accumulate_res = collections.defaultdict(list)
    for i, data in enumerate(iter(dataloader)):
        if (i+1) % args.log_freq == 0:
            for k, v in accumulate_res.items():
                v = np.array(v).mean(0)
                print(f'{k}: {v}')

        pbar.update(1)
        inp_B3HW, captions = data
        tokens = text_tokenizer(text=captions, max_length=text_tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt')  # todo: put this into dataset
        input_ids = tokens.input_ids.cuda(non_blocking=True)
        mask = tokens.attention_mask.cuda(non_blocking=True)
        text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
        lens: List[int] = mask.sum(dim=-1).tolist()
        cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
        Ltext = max(lens)
        kv_compact = []
        for len_i, feat_i in zip(lens, text_features.unbind(0)):
            kv_compact.append(feat_i[:len_i])
        kv_compact = torch.cat(kv_compact, dim=0)
        text_cond_tuple: Tuple[torch.FloatTensor, List[int], torch.LongTensor, int] = (kv_compact, lens, cu_seqlens_k, Ltext)

        h_div_w = inp_B3HW.shape[-2] / inp_B3HW.shape[-1]
        h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
        raw_last_l = np.array(scale_schedule[-1]).prod()
        
        # [prepare]
        B = inp_B3HW.shape[0] if isinstance(inp_B3HW, torch.Tensor) else inp_B3HW[0].shape[0]
        V = vae.vocab_size
        
        # [forward]
        with torch.amp.autocast('cuda', enabled=False):
            with torch.no_grad():
                if args.apply_spatial_patchify:
                    vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
                else:
                    vae_scale_schedule = scale_schedule
                raw_features, _, _ = vae.encode_for_raw_features(inp_B3HW.to(device), scale_schedule=vae_scale_schedule)
            
            x_BLC_wo_prefix, gt_ms_idx_Bl = bitwise_self_correction.flip_requant(vae_scale_schedule, inp_B3HW, raw_features, device)
            training_seq_len = np.array(scale_schedule).prod(axis=1).sum()
            x_BLC_wo_prefix = x_BLC_wo_prefix[:, :(training_seq_len-np.array(scale_schedule[0]).prod()), :]

            with torch.no_grad():
                logits_BLV = infinity(text_cond_tuple, x_BLC_wo_prefix, scale_schedule=scale_schedule) # [bs, 1*1+...+64*64, vocab_size or log2(vocab_size)*2]
            
            if args.vis_model_flop_param:
                from torchinfo import summary
                res = summary(infinity, input_data=(text_cond_tuple, x_BLC_wo_prefix, scale_schedule))
                print(res)

            batch_size, seq_len = logits_BLV.shape[:2]
            seq_len_each = [idx_Bl.shape[1] for idx_Bl in gt_ms_idx_Bl]
            
            gt_BL = torch.cat(gt_ms_idx_Bl, dim=1)[:,:training_seq_len].contiguous().type(torch.long) # [bs, 1*1+...+64*64, 16] or [bs, 1*1+...+64*64]
            tmp_bs, tmp_seq_len, tmp_channel = logits_BLV.shape
            assert tmp_channel == vae.codebook_dim * 2
            res_loss = torch.nn.functional.cross_entropy(logits_BLV.reshape(tmp_bs, tmp_seq_len, vae.codebook_dim, 2).permute(0,3,1,2), gt_BL, reduction='none')
            res_loss = res_loss.mean(dim=-1).mean(0)

            if args.reweight_loss_by_scale:
                lw = []
                last_scale_area = np.sqrt(np.array(scale_schedule[-1]).prod())
                for (ph, pw) in scale_schedule:
                    this_scale_area = np.sqrt(ph * pw)
                    lw.extend([last_scale_area / this_scale_area for _ in range(ph * pw)])
                lw = torch.tensor(lw, device=device)
                lw = lw / lw.sum()
            else:
                lw = 1. / training_seq_len
            loss_reweight_by_scale = res_loss.mul(lw).sum(dim=-1).mean().item()

            bitwise_acc = (logits_BLV.reshape(B, seq_len, vae.codebook_dim, 2).argmax(dim=-1) == gt_BL).float() # shape: [bs, seq_len, codebook_dim]
            res_bit_acc = bitwise_acc.mean(-1).mean(0)
            res_token_acc = (bitwise_acc.sum(-1) == vae.codebook_dim).float().mean(0)
            loss_mean, acc_bit_mean, acc_token_mean = res_loss.mean().item(), res_bit_acc.mean().item() * 100., res_token_acc.mean().item() * 100.
            ptr = 0
            L_list, acc_bit_list, acc_token_list = [], [], []
            for scale_ind in range(len(scale_schedule)):
                start, end = ptr, ptr + np.array(scale_schedule[scale_ind]).prod()
                L_list.append(res_loss[start:end].mean().item())
                acc_bit_list.append(res_bit_acc[start:end].mean().item() * 100.)
                acc_token_list.append(res_token_acc[start:end].mean().item() * 100.)
                ptr = end
            accumulate_res['loss_bit_mean'].append(loss_mean)
            accumulate_res['acc_bit_mean'].append(acc_bit_mean)
            accumulate_res['acc_token_mean'].append(acc_token_mean)
            accumulate_res['loss_reweight_by_scale'].append(loss_reweight_by_scale)
            accumulate_res['loss_by_scale'].append(L_list)
            accumulate_res['acc_bit_list_by_scale'].append(acc_bit_list)
            accumulate_res['acc_token_list_by_scale'].append(acc_token_list)
    
    for k, v in accumulate_res.items():
        if len(np.array(v).shape) == 1:
            v = np.array(v).mean(0)
        else:
            v = np.array(v).mean(0).tolist()
        accumulate_res[k] = v
        print(f'{k}: {v}')
    
    save_file = osp.join(args.save_dir, 'val_res.json')
    os.makedirs(osp.dirname(save_file), exist_ok=True)
    with open(save_file, 'w') as f:
        json.dump(accumulate_res, f, indent=2)
    print(f'Save val results to {save_file}')
