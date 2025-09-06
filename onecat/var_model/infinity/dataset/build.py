import datetime
import os
import os.path as osp
import random
import subprocess
from functools import partial
from typing import Optional
import time

import pytz

from vargpt_qwen_v1_1.var_model.infinity.dataset.dataset_t2i_iterable import T2IIterableDataset

try:
    from grp import getgrgid
    from pwd import getpwuid
except:
    pass
import PIL.Image as PImage
from PIL import ImageFile
import numpy as np
from torchvision.transforms import transforms
from torchvision.transforms.functional import resize, to_tensor
import torch.distributed as tdist

from torchvision.transforms import InterpolationMode
bicubic = InterpolationMode.BICUBIC
lanczos = InterpolationMode.LANCZOS
PImage.MAX_IMAGE_PIXELS = (1024 * 1024 * 1024 // 4 // 3) * 5
ImageFile.LOAD_TRUNCATED_IMAGES = False


def time_str(fmt='[%m-%d %H:%M:%S]'):
    return datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(fmt)


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


def denormalize_pm1_into_01(x):  # denormalize x from [-1, 1] to [0, 1]
    return x.add(1).mul_(0.5)


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=PImage.BOX
        )
    
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=PImage.LANCZOS
    )
    
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return PImage.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class RandomResize:
    def __init__(self, mid_reso, final_reso, interpolation):
        ub = max(round((mid_reso + (mid_reso-final_reso) / 8) / 4) * 4, mid_reso)
        self.reso_lb, self.reso_ub = final_reso, ub
        self.interpolation = interpolation
    
    def __call__(self, img):
        return resize(img, size=random.randint(self.reso_lb, self.reso_ub), interpolation=self.interpolation)
    
    def __repr__(self):
        return f'RandomResize(reso=({self.reso_lb}, {self.reso_ub}), interpolation={self.interpolation})'


def load_save(reso=512):
    import os
    from PIL import Image as PImage
    from torchvision.transforms import transforms, InterpolationMode
    aug = transforms.Compose([
        transforms.Resize(512, interpolation=InterpolationMode.LANCZOS),
        transforms.CenterCrop((512, 512))
    ])
    src_folder = r'C:\Users\16333\Pictures\imgs_to_visual_v2'
    ls = [os.path.join(src_folder, x) for x in ('1.jpg', '2.jpg', '3.png', '4.png', '5.png')]
    print(ls)
    imgs = []
    for i, fname in enumerate(ls):
        assert os.path.exists(fname)
        with PImage.open(fname) as img:
            img = img.convert('RGB')
            img = aug(img)
            imgs.append(img)
        dst_d, dst_f = os.path.split(fname)
        dst = os.path.join(dst_d, f'crop{dst_f.replace(".jpg", ".png")}')
        img.save(dst)
    
    W, H = imgs[0].size
    WW = W * len(imgs)
    new_im = PImage.new('RGB', (WW, H))
    x_offset = 0
    for img in imgs:
        new_im.paste(img, (x_offset, 0))
        x_offset += W
    dst = os.path.join(src_folder, f'junfeng.png')
    new_im.save(dst)


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')


def build_t2i_dataset(
    args,
    data_path: str,
    data_load_reso: int,
    max_caption_len: int,
    short_prob=0.2,
    load_vae_instead_of_image=False
):
    if args.use_streaming_dataset:
        return T2IIterableDataset(
            data_path, 
            max_caption_len=max_caption_len, 
            short_prob=short_prob, 
            load_vae_instead_of_image=load_vae_instead_of_image, 
            buffersize=args.iterable_data_buffersize,
            pn=args.pn,
            online_t5=args.online_t5,
            batch_size=args.batch_size,
            # num_replicas=tdist.get_world_size(), # 1,
            # rank=tdist.get_rank(), # 0
            num_replicas= 1,
            rank= 0,
            dataloader_workers=args.workers,
            dynamic_resolution_across_gpus=args.dynamic_resolution_across_gpus,
            enable_dynamic_length_prompt=args.enable_dynamic_length_prompt,
            seed=args.seed if args.seed is not None else int(time.time()),
        )
    else:
        raise ValueError(f'args.use_streaming_dataset={args.use_streaming_dataset} unsupported')


def pil_load(path: str, proposal_size):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f)
        w: int = img.width
        h: int = img.height
        sh: int = min(h, w)
        if sh > proposal_size:
            ratio: float = proposal_size / sh
            w = round(ratio * w)
            h = round(ratio * h)
        img.draft('RGB', (w, h))
        img = img.convert('RGB')
    return img


def rewrite(im: PImage, file: str, info: str):
    kw = dict(quality=100)
    if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
        kw['compression'] = 'none'
    elif file.lower().endswith('.webp'):
        kw['lossless'] = True
    
    st = os.stat(file)
    uname = getpwuid(st.st_uid).pw_name
    gname = getgrgid(st.st_gid).gr_name
    mode = oct(st.st_mode)[-3:]
    
    local_file = osp.basename(file)
    im.save(local_file, **kw)
    print(f'************* <REWRITE: {info}> *************  @  {file}')
    subprocess.call(f'sudo mv {local_file} {file}; sudo chown {uname}:{gname} {file}; sudo chmod {mode} {file}', shell=True)
