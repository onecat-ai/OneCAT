import math
import os
import signal
import sys
import time
from typing import List, Optional, Tuple, Union

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from memory_profiler import profile

import infinity.utils.dist as dist
from vargpt_qwen_v1_1.var_model.infinity.utils import misc

class NullCtx:
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def handle_timeout(signum, frame):
    raise TimeoutError('took too long')


def per_param_clip_grad_norm_(parameters, thresh: float, stable=False, fp=None) -> (float, float):
    skipped, max_grad = [], 0
    for pi, p in enumerate(parameters):
        if p.grad is not None:
            g = p.grad.data.norm(2).item() + 1e-7
            max_grad = max(max_grad, g)
            clip_coef = thresh / g
            if clip_coef < 1:
                if stable and clip_coef < 0.2:
                    skipped.append(clip_coef)
                    p.grad.data.mul_(0)     # todo NOTE: inf.mul_(0)==nan will shrink the scale ratio, but inf.zero_()==0 won't
                else:
                    p.grad.data.mul_(clip_coef)
    
    # if fp is not None: fp.write(f'[per_param_clip_grad_norm_:47] finished.\n'); fp.flush()
    return 0 if len(skipped) == 0 else math.log10(max(min(skipped), 1e-7)), max_grad


class AmpOptimizer:
    def __init__(
        self,
        model_name_3letters: str, mixed_precision: int,
        optimizer: torch.optim.Optimizer, model_maybe_fsdp: Union[torch.nn.Module, FSDP],
        r_accu: float, grad_clip: float, zero: int,
    ):
        self.enable_amp = mixed_precision > 0
        self.zero = zero
        if self.enable_amp:
            self.using_fp16_rather_bf16 = mixed_precision != 2
            self.max_sc = float(mixed_precision if mixed_precision > 128 else 32768)
            
            # todo: on both V100 and A100, torch.get_autocast_gpu_dtype() returns fp16, not bf16.
            self.amp_ctx = torch.autocast('cuda', enabled=True, dtype=torch.float16 if self.using_fp16_rather_bf16 else torch.bfloat16, cache_enabled=self.zero == 0)    # todo: cache_enabled=False
            if self.using_fp16_rather_bf16:
                self.scaler = torch.cuda.amp.GradScaler(init_scale=2. ** 11, growth_interval=1000)
            else:
                self.scaler = None
        else:
            self.using_fp16_rather_bf16 = True
            self.amp_ctx = NullCtx()
            self.scaler = None
        
        t = torch.zeros(dist.get_world_size())
        t[dist.get_rank()] = float(self.enable_amp)
        dist.allreduce(t)
        assert round(t.sum().item()) in {0, dist.get_world_size()}, f'enable_amp: {t}'
        
        t = torch.zeros(dist.get_world_size())
        t[dist.get_rank()] = float(self.using_fp16_rather_bf16)
        dist.allreduce(t)
        assert round(t.sum().item()) in {0, dist.get_world_size()}, f'using_fp16_rather_bf16: {t}'
        
        self.model_name_3letters = model_name_3letters
        self.optimizer, self.model_maybe_fsdp = optimizer, model_maybe_fsdp
        self.r_accu = r_accu
        
        self.paras = self.names = ...    # todo: solve EMA-related codes
        
        self.grad_clip, self.grad_clip_we = grad_clip, 0    # todo: disable wclip
        if self.grad_clip > 100:
            self.grad_clip %= 100
            self.per_param = True
        else:
            self.per_param = False
        self.per_param = False          # todo: disable wclip
        
        self.early_clipping = grad_clip > 0 and not hasattr(optimizer, 'global_grad_norm')
        self.late_clipping = grad_clip > 0 and hasattr(optimizer, 'global_grad_norm')   # deepspeed's optimizer
        
        self.fp = None
        self.last_orig_norm: torch.Tensor = torch.tensor(0.1)
    
    @torch.no_grad()
    def log_param(self, ep: int):
        if self.zero == 0:
            for name, values in get_param_for_log(self.model_name_3letters, self.model_maybe_fsdp.named_parameters()).items():
                values: List[float]
                if len(values) == 1:    # e.g., cls token will only have one value
                    values.append(values[0])
        else:
            ...
            # todo: log params
    
    # @profile(precision=4, stream=open('amp_sc.log', 'w+'))
    def backward_clip_step(
        self, ep: int, it: int, g_it: int, stepping: bool, logging_params: bool, loss: torch.Tensor, clip_decay_ratio=1, stable=False,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        # backward
        loss = loss.mul(self.r_accu)   # r_accu == 1.0 / n_gradient_accumulation
        orig_norm = scaler_sc = None
        # if self.fp is not None:
        #     if g_it % 20 == 0: self.fp.seek(0); self.fp.truncate(0)
        if self.scaler is not None:
            self.scaler.scale(loss).backward(retain_graph=False, create_graph=False)  # retain_graph=retain_graph, create_graph=create_graph
        else:
            loss.backward(retain_graph=False, create_graph=False)
        # if self.fp is not None: self.fp.write(f'[backward_clip_step:131] [it{it}, g_it{g_it}] after backward\n'); self.fp.flush()
        
        # clip gradients then step optimizer
        if stepping:
            if self.scaler is not None: self.scaler.unscale_(self.optimizer)    # now the gradient can be correctly got
            # if self.fp is not None: self.fp.write(f'[backward_clip_step:137] [it{it}, g_it{g_it}] after scaler.unscale_\n'); self.fp.flush()
            
            skipped, orig_norm = 0, self.last_orig_norm
            # try:
            if self.fp is not None:
                if g_it % 10 == 0: self.fp.seek(0); self.fp.truncate(0)
                self.fp.write(f'<ep{ep} it{it} {g_it}>\n'); self.fp.flush()
            if self.early_clipping:
                c = self.grad_clip * clip_decay_ratio
                if self.zero:
                    orig_norm: Optional[torch.Tensor] = self.model_maybe_fsdp.clip_grad_norm_(c)
                else:
                    orig_norm: Optional[torch.Tensor] = torch.nn.utils.clip_grad_norm_(self.model_maybe_fsdp.parameters(), c)
            
            # if self.fp is not None: self.fp.write(f'[backward_clip_step:175] [it{it}, g_it{g_it}] before opt step\n'); self.fp.flush()
            if self.scaler is not None:
                self.scaler: torch.cuda.amp.GradScaler
                if self.zero:
                    # synchronize found_inf_per_device before calling step, so that even if only some ranks found inf on their sharded params, all other ranks will know
                    # otherwise, when saving FSDP optimizer state, it will cause AssertionError saying "Different ranks have different values for step."
                    for optimizer_state in self.scaler._per_optimizer_states.values():
                        for t in optimizer_state['found_inf_per_device'].values():
                            dist.allreduce(t)   # ideally, each rank only has one single t; so no need to use async allreduce
                
                self.scaler.step(self.optimizer)
                scaler_sc: Optional[float] = self.scaler.get_scale()
                if scaler_sc > self.max_sc: # fp16 will overflow when >65536, so multiply 32768 could be dangerous
                    # print(f'[fp16 scaling] too large loss scale {scaler_sc}! (clip to {self.max_sc:g})')
                    self.scaler.update(new_scale=self.max_sc)
                else:
                    self.scaler.update()
                try:
                    scaler_sc = float(math.log2(scaler_sc))
                except Exception as e:
                    print(f'[scaler_sc = {scaler_sc}]\n' * 15, flush=True)
                    time.sleep(1)
                    print(f'[scaler_sc = {scaler_sc}]\n' * 15, flush=True)
                    raise e
            else:
                self.optimizer.step()
            
            if self.late_clipping:
                orig_norm: Optional[torch.Tensor] = self.optimizer.global_grad_norm
            self.last_orig_norm = orig_norm
            # no zero_grad calling here, gonna log those gradients!
        return orig_norm, scaler_sc
    
    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict()
        } if self.scaler is None else {
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state, strict=True):
        if self.scaler is not None:
            try: self.scaler.load_state_dict(state['scaler'])
            except Exception as e: print(f'[fp16 load_state_dict err] {e}')
        self.optimizer.load_state_dict(state['optimizer'])
