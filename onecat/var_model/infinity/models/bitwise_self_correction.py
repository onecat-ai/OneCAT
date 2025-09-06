import os
import os.path as osp

import torch
import torch.nn.functional as F
import numpy as np


def labels2image(all_indices, label_type='int_label', scale_schedule=None):
    summed_codes, recons_imgs = self.vae.decode_from_indices(all_indices, scale_schedule, label_type)
    recons_img = recons_imgs[0]
    recons_img = (recons_img + 1) / 2
    recons_img = recons_img.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)[:,:,::-1]
    return recons_img

def features2image(raw_features):
    recons_imgs = self.vae.decode(raw_features.squeeze(-3))
    recons_img = recons_imgs[0]
    recons_img = (recons_img + 1) / 2
    recons_img = recons_img.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)[:,:,::-1]
    return recons_img

class BitwiseSelfCorrection(object):
    def __init__(self, vae):
        self.noise_apply_layers = 13
        self.noise_apply_requant = 1
        self.noise_apply_strength = 0.3
        self.apply_spatial_patchify = 0
        self.vae = vae
        self.debug_bsc = False

    def flip_requant(self, vae_scale_schedule, inp_B3HW, raw_features, device):
        # with torch.amp.autocast('cuda', enabled = False):
        B = raw_features.shape[0]
        if raw_features.dim() == 4:
            codes_out = raw_features.unsqueeze(2)
        else:
            codes_out = raw_features
        cum_var_input = 0
        gt_all_bit_indices = []
        pred_all_bit_indices = []
        x_BLC_wo_prefix = []
        for si, (pt, ph, pw) in enumerate(vae_scale_schedule):
            residual = codes_out - cum_var_input
            if si != len(vae_scale_schedule)-1:
                residual = F.interpolate(residual, size=vae_scale_schedule[si], mode=self.vae.quantizer.z_interplote_down).contiguous()
            quantized, _, bit_indices, loss = self.vae.quantizer.lfq(residual) # quantized shape: [B, d_vae, 1, h, w], bit_indices shape: [B,1,h,w,d_vae]
            gt_all_bit_indices.append(bit_indices)
            if si < self.noise_apply_layers:
                noise_apply_strength = np.random.randint(0, 100 * self.noise_apply_strength+1) * 0.01
                mask = torch.rand(*bit_indices.shape).to(device) < noise_apply_strength
                pred_bit_indices = bit_indices.clone()
                pred_bit_indices[mask] = 1 - pred_bit_indices[mask]
                pred_all_bit_indices.append(pred_bit_indices)
                if self.noise_apply_requant:
                    quantized = self.vae.quantizer.lfq.indices_to_codes(pred_bit_indices, label_type = 'bit_label')
            else:
                pred_all_bit_indices.append(bit_indices)
            cum_var_input = cum_var_input + F.interpolate(quantized, size=vae_scale_schedule[-1], mode=self.vae.quantizer.z_interplote_up).contiguous()
            if si < len(vae_scale_schedule)-1:
                this_scale_input = F.interpolate(cum_var_input, size=vae_scale_schedule[si+1], mode=self.vae.quantizer.z_interplote_up).contiguous()
                if self.apply_spatial_patchify:
                    # (B,d,1,H,W) -> (B,d,H,W) -> (B,4d,H/2,W/2)
                    this_scale_input = torch.nn.functional.pixel_unshuffle(this_scale_input.squeeze(-3), 2)
                x_BLC_wo_prefix.append(this_scale_input.reshape(*this_scale_input.shape[:2], -1).permute(0,2,1)) # (B,H/2*W/2,4C) or (B,H*W,C)

        if self.apply_spatial_patchify:
            gt_ms_idx_Bl = []
            for item in gt_all_bit_indices:
                # item shape: (B,1,H,W,d)
                item = item.squeeze(1).permute(0,3,1,2) # (B,d,H,W)
                # (B,d,H,W) -> (B,4d,H/2,W/2)
                item = torch.nn.functional.pixel_unshuffle(item, 2)
                # (B,4d,H/2,W/2) -> (B,H/2,W/2,4d) -> (B,H/2*w/2,4d)
                item = item.permute(0,2,3,1).reshape(B, -1, 4*self.vae.codebook_dim)
                gt_ms_idx_Bl.append(item)
        else:
            gt_ms_idx_Bl = [item.reshape(B, -1, self.vae.codebook_dim) for item in gt_all_bit_indices]
        x_BLC_wo_prefix = torch.cat(x_BLC_wo_prefix, 1)

        if self.debug_bsc:
            self.visualize(vae_scale_schedule, inp_B3HW, gt_all_bit_indices, pred_all_bit_indices)
    
        return x_BLC_wo_prefix, gt_ms_idx_Bl
    
    def visualize(self, vae_scale_schedule, inp_B3HW, gt_all_bit_indices, pred_all_bit_indices):
        gt_img = (inp_B3HW.squeeze(-3) + 1) / 2 * 255
        gt_img = gt_img[0].permute(1,2,0).cpu().numpy().astype(np.uint8)[:,:,::-1]
        recons_img_2 = labels2image(gt_all_bit_indices, label_type='bit_label', scale_schedule=vae_scale_schedule)
        recons_img_3 = labels2image(pred_all_bit_indices, label_type='bit_label', scale_schedule=vae_scale_schedule)
        cat_image = np.concatenate([gt_img, recons_img_2, recons_img_3], axis=1)
        save_path = osp.abspath('non_teacher_force.jpg')
        cv2.imwrite(save_path, cat_image)
        print(f'Save to {save_path}')
        import pdb; pdb.set_trace()
        print(cat_image.shape)
        