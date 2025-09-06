import glob
import os
import pickle
import random
import re
import time
from functools import partial
from os import path as osp
from typing import List, Tuple, Union
import json
import itertools
import concurrent.futures
from multiprocessing import cpu_count

import tqdm
import numpy as np
import torch
import pandas as pd
from PIL import Image as PImage
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as tdist

from vargpt_qwen_v1_1.var_model.infinity.utils.dynamic_resolution import dynamic_resolution_h_w, get_h_div_w_template2indices, h_div_w_templates
from vargpt_qwen_v1_1.var_model.infinity.utils.large_file_util import get_part_jsonls, split_large_txt_files


def center_crop_to_tensor_pm1(pil_image, mid_reso: int, final_reso: int):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    Then to_tensor and normalize to [-1, 1]
    """
    while min(*pil_image.size) >= 2 * mid_reso:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=PImage.BOX
        )
    
    if mid_reso == final_reso == pil_image.size[0] == pil_image.size[1]:
        im = to_tensor(pil_image)
    else:
        # resize the shorter edge to mid_reso
        scale = mid_reso / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=PImage.LANCZOS
        )
        
        # crop the center out
        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - final_reso) // 2
        crop_x = (arr.shape[1] - final_reso) // 2
        # return PImage.fromarray(arr[crop_y: crop_y + final_reso, crop_x: crop_x + final_reso])
        im = to_tensor(arr[crop_y: crop_y + final_reso, crop_x: crop_x + final_reso])
    
    return im.add(im).add_(-1)

def transform(pil_img, tgt_h, tgt_w):
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    # print(f'im size {im.shape}')
    return im.add(im).add_(-1)

def process_short_text(short_text):
    if '--' in short_text:
        processed_text = short_text.split('--')[0]
        if processed_text:
            short_text = processed_text
    return short_text


class T2IIterableDataset(IterableDataset):
    def __init__(
        self, 
        meta_folder: str, 
        max_caption_len=512, 
        short_prob=0.2, 
        load_vae_instead_of_image=False,
        buffersize: int = 10000,
        seed: int = 0, 
        pn: str = '',
        online_t5: bool = True,
        batch_size: int = 2,
        num_replicas: int = 1, # 1,
        rank: int = 0, # 0
        dataloader_workers: int = 2,
        dynamic_resolution_across_gpus: bool = True,
        enable_dynamic_length_prompt: bool = True,
        **kwargs,
    ):
        self.meta_folder = meta_folder
        self.pn = pn
        self.online_t5 = online_t5
        self.buffer_size = buffersize
        self.num_replicas = num_replicas
        self.rank = rank
        self.worker_id = 0
        self.global_worker_id = 0
        self.dataloader_workers = max(1, dataloader_workers)
        self.max_caption_len = max_caption_len
        self.short_prob = short_prob
        self.load_vae_instead_of_image = load_vae_instead_of_image # set to false
        self.dynamic_resolution_across_gpus = dynamic_resolution_across_gpus
        self.enable_dynamic_length_prompt = enable_dynamic_length_prompt
        self.batch_size = batch_size
        print(f'self.dynamic_resolution_across_gpus: {self.dynamic_resolution_across_gpus}')
        print(f'self.enable_dynamic_length_prompt: {self.enable_dynamic_length_prompt}')
        print(f'self.buffer_size: {self.buffer_size}')
        self.shuffle = True
        self.global_workers = self.num_replicas * self.dataloader_workers
        self.h_div_w_template2generator, self.samples_div_gpus_workers_batchsize_2batches, total_samples = self.set_h_div_w_template2generator()
        self.split_meta_files()
        self.seed = seed
        self.epoch_worker_generator = None
        self.epoch_global_worker_generator = None
        self.set_epoch(0)
        print(f'num_replicas: {num_replicas}, rank: {rank}, dataloader_workers: {dataloader_workers}, seed:{seed}, samples_div_gpus_workers_batchsize_2batches: {self.samples_div_gpus_workers_batchsize_2batches}')

    def set_h_div_w_template2generator(self,):
        samples_div_gpus_workers_batchsize_2batches = 0
        h_div_w_template2generator = {}
        total_samples = 0
        for filepath in sorted(glob.glob(osp.join(self.meta_folder, '*.jsonl'))):
            filename = osp.basename(filepath)
            h_div_w_template, num_of_samples = osp.splitext(filename)[0].split('_')
            total_samples += int(num_of_samples)
        for filepath in sorted(glob.glob(osp.join(self.meta_folder, '*.jsonl'))):
            filename = osp.basename(filepath)
            h_div_w_template, num_of_samples = osp.splitext(filename)[0].split('_')
            num_of_samples = int(num_of_samples)
            if num_of_samples < self.global_workers:
                print(f'{filepath} has too few examples ({num_of_samples}, proportion: {num_of_samples/total_samples*100:.1f}%), < global workers ({self.global_workers})! Skip h_div_w_template: {h_div_w_template}')
                continue
            print(f'{filepath} has sufficient examples ({num_of_samples}), proportion: {num_of_samples/total_samples*100:.1f}%, > global workers ({self.global_workers})! Preserve h_div_w_template: {h_div_w_template}')
            num_of_batches = max(1, int((num_of_samples // self.global_workers // self.batch_size)))
            h_div_w_template2generator[h_div_w_template] = {
                'filepath': filepath,
                'num_of_samples': num_of_samples,
                'num_of_batches': num_of_batches,
            }
            samples_div_gpus_workers_batchsize_2batches += num_of_batches
        return h_div_w_template2generator, samples_div_gpus_workers_batchsize_2batches, total_samples

    def split_meta_files(self, ):
        print('[data preprocess] split_meta_files')

        def split_and_sleep(generator_info):
            missing, chunk_id2save_files = get_part_jsonls(generator_info['filepath'], generator_info['num_of_samples'], parts=self.num_replicas)
            if missing:
                tdist.barrier()
                if self.rank == 0:
                    split_large_txt_files(generator_info['filepath'], chunk_id2save_files)
                else:
                    sleep_time = int(generator_info['num_of_samples'] / 30000000 * 10)
                    print(f'[data preprocess] sleep {sleep_time} minutes awaiting rank0 split_meta_files...')
                    time.sleep(sleep_time*60)
                tdist.barrier()
            generator_info['part_filepaths'] = sorted(list(chunk_id2save_files.values()))
            return generator_info

        with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            futures = {executor.submit(split_and_sleep, generator_info): h_div_w_template for h_div_w_template, generator_info in self.h_div_w_template2generator.items()}
            for future in concurrent.futures.as_completed(futures):
                h_div_w_template = futures[future]
                try:
                    self.h_div_w_template2generator[h_div_w_template] = future.result()
                except Exception as exc:
                    print(f'[data preprocess] h_div_w_template {h_div_w_template} generated an exception: {exc}')

        print('[data preprocess] split_meta_files done')

    def set_global_worker_id(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            worker_total_num = worker_info.num_workers
            worker_id = worker_info.id
        else:
            worker_id = 0
            worker_total_num = 1
        assert worker_total_num == self.dataloader_workers, print(worker_total_num, self.dataloader_workers)
        self.worker_id = worker_id
        self.global_worker_id = self.rank * self.dataloader_workers + worker_id
        # print(f'Set worker_id to {self.worker_id}, global_worker_id to {self.global_worker_id}')
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        self.set_generator()
    
    def set_generator(self, ):
        self.epoch_worker_generator = np.random.default_rng(self.seed + self.epoch + self.worker_id)
        self.epoch_global_worker_generator = np.random.default_rng(self.seed + self.epoch + self.global_worker_id)
    
    def get_h_div_w_template_2_unlearned_batches(self,):
        h_div_w_template_2_unlearned_batches = {}
        total_unlearned_batches = 0
        for h_div_w_template, generator_info in self.h_div_w_template2generator.items():
            h_div_w_template_2_unlearned_batches[h_div_w_template] = generator_info['num_of_batches']
            total_unlearned_batches += generator_info['num_of_batches']
        self.total_unlearned_batches = total_unlearned_batches
        self.h_div_w_template_2_unlearned_batches = h_div_w_template_2_unlearned_batches
        assert self.total_unlearned_batches == self.samples_div_gpus_workers_batchsize_2batches

    def _next_h_div_w_template(self,):
        while True:
            self.get_h_div_w_template_2_unlearned_batches()
            while self.total_unlearned_batches > 0:
                if self.dynamic_resolution_across_gpus:
                    i = self.epoch_global_worker_generator.integers(0, self.total_unlearned_batches)
                else:
                    i = self.epoch_worker_generator.integers(0, self.total_unlearned_batches)
                self.total_unlearned_batches -= 1
                for h_div_w_template, unlearned_batches in self.h_div_w_template_2_unlearned_batches.items():
                    if i < unlearned_batches:
                        yield h_div_w_template
                        self.h_div_w_template_2_unlearned_batches[h_div_w_template] -= 1
                        break
                    else:
                        i -= unlearned_batches

    def __iter__(self):
        self.set_global_worker_id()
        self.set_generator()
        
        for h_div_w_template, generator_info in self.h_div_w_template2generator.items():
            proportion = generator_info['num_of_batches'] / self.samples_div_gpus_workers_batchsize_2batches
            h_div_w_buffer_size = int(self.buffer_size * proportion)
            h_div_w_buffer_size = min(max(1, h_div_w_buffer_size), generator_info['num_of_batches'] * self.batch_size)
            if 'mem_buffer' in generator_info:
                del generator_info['mem_buffer']
            mem_buffer = []
            for _ in range(h_div_w_buffer_size):
                mem_buffer.append(self.infinite_next(generator_info))
            generator_info['mem_buffer'] = mem_buffer
        
        next_h_div_w_template_iter = self._next_h_div_w_template()
        # while True:
        for _ in range(self.samples_div_gpus_workers_batchsize_2batches):
            batch_data = []
            h_div_w_template = next(next_h_div_w_template_iter)
            while len(batch_data) < self.batch_size:
                try:
                    generator_info = self.h_div_w_template2generator[h_div_w_template]
                    mem_buffer = generator_info['mem_buffer']
                    i = self.epoch_global_worker_generator.integers(0, len(mem_buffer))
                    data_item = mem_buffer[i]
                    mem_buffer[i] = self.infinite_next(generator_info)
                    ret, model_input = self.prepare_model_input(json.loads(data_item)) # data_item[0] is row number of panda dataframe
                    if ret:
                        c_, h_, w_ = model_input[1].shape[-3:]
                        if c_ != 3 or np.abs(h_/w_-float(h_div_w_template)) > 0.01:
                            print(f'Croupt data item: {data_item}')
                        else:
                            batch_data.append(model_input)
                    del data_item
                except Exception as e:
                    print(e)
            captions = [item[0] for item in batch_data]
            images = torch.stack([item[1] for item in batch_data])
            yield (images, captions)
            del batch_data
            del images
            del captions
    
    def infinite_next(self, generator_info):
        try:
            if 'sub_iterator' not in generator_info:
                raise StopIteration
            return next(generator_info['sub_iterator'])
        except StopIteration as e:
            if 'record_iterator' in generator_info:
                generator_info['record_iterator'].close()
            if 'sub_iterator' in generator_info:
                del generator_info['sub_iterator']
            part_filepath = generator_info['part_filepaths'][self.rank]
            generator_info['record_iterator'] = open(part_filepath, 'r')
            part_num_of_samples = int(osp.splitext(osp.basename(part_filepath))[0].split('_')[-1])
            # print(f'part_filepath: {part_filepath}, rank: {self.rank}, worker_id:{self.worker_id}, part_num_of_samples: {part_num_of_samples}, dataloader_workers: {self.dataloader_workers}')
            generator_info['sub_iterator'] = itertools.islice(generator_info['record_iterator'], self.worker_id, part_num_of_samples, self.dataloader_workers)
            return next(generator_info['sub_iterator'])

    def __len__(self):
        return self.samples_div_gpus_workers_batchsize_2batches * self.dataloader_workers
    
    def total_samples(self):
        return self.samples_div_gpus_workers_batchsize_2batches * self.dataloader_workers * self.num_replicas * self.batch_size

    def get_text_input(self, long_text_input, short_text_input, long_text_type):
        random_value = self.epoch_global_worker_generator.random()
        if self.enable_dynamic_length_prompt and long_text_type != 'user_prompt':
            long_text_elems = [item for item in long_text_input.split('.') if item]
            if len(long_text_elems):
                first_sentence_words = [item for item in long_text_elems[0].split(' ') if item]
            else:
                first_sentence_words = 0
            if len(first_sentence_words) >= 15:
                num_sentence4short_text = 1
            else:
                num_sentence4short_text = 2
            if not short_text_input:
                short_text_input = '.'.join(long_text_elems[:num_sentence4short_text])
            if random_value < self.short_prob:
                return short_text_input
            if len(long_text_elems) <= num_sentence4short_text:
                return long_text_input
            select_sentence_num = self.epoch_global_worker_generator.integers(num_sentence4short_text+1, len(long_text_elems)+1)
            return '.'.join(long_text_elems[:select_sentence_num])
        else:
            if short_text_input and random_value < self.short_prob:
                return short_text_input
            return long_text_input

    def prepare_model_input(self, data_item) -> Tuple:
        img_path, h_div_w = data_item['image_path'], data_item['h_div_w']
        short_text_input, long_text_input = data_item['text'], data_item['long_caption']
        long_text_type = data_item.get('long_caption_type', 'user_prompt')
        text_input = self.get_text_input(long_text_input, short_text_input, long_text_type)
        text_input = process_short_text(text_input)

        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w - h_div_w_templates))]
        try:
            if self.load_vae_instead_of_image:
                img_B3HW = None
                vae_path = self.get_vae_path(img_path)
                with open(vae_path, 'rb') as f:
                    gt_ms_idx_Bl = pickle.load(f)
            else:
                gt_ms_idx_Bl = None
                with open(img_path, 'rb') as f:
                    img: PImage.Image = PImage.open(f)
                    img = img.convert('RGB')
                    tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][self.pn]['pixel']
                    img_B3HW = transform(img, tgt_h, tgt_w)
            if not self.online_t5:
                short_t5_path, long_t5_path = self.get_t5_path(img_path)
                if self.epoch_global_worker_generator.random() <= self.short_prob:
                    t5_path = short_t5_path
                else:
                    t5_path = long_t5_path
                t5_meta = np.load(t5_path)
                text_input = t5_meta['t5_feat'][:self.max_caption_len] # L x C
        except Exception as e:
            print(f'input error: {e}, skip to another index')
            return False, None

        if self.load_vae_instead_of_image:
            return True, (text_input, *gt_ms_idx_Bl)
        else:
            return True, (text_input, img_B3HW)

    @staticmethod
    def collate_function(batch, online_t5: bool = False) -> None:
        pass

if __name__ == '__main__':
    # torchrun --nnodes=1 --nproc-per-node=2 --master_addr=$METIS_WORKER_0_HOST --master_port=$METIS_WORKER_0_PORT dataset/dataset_t2i_iterable.py
    tdist.init_process_group(backend='nccl')
    batch_size = 2
    dataloader_workers = 12
    dataset = T2IIterableDataset(
        args=None, 
        meta_folder='data/train_splits/xxx_pretrain/jsonl_files_filter_duplicate_captions',
        data_load_reso=None, 
        max_caption_len=512, 
        short_prob=1.0, 
        load_vae_instead_of_image=False,
        buffersize=100000,
        seed=0, 
        online_t5=True,
        pn='0.06M',
        batch_size=batch_size,
        num_replicas=8, # tdist.get_world_size(),
        rank=tdist.get_rank(), # 0
        dataloader_workers=dataloader_workers,
    )
    dataloader = DataLoader(dataset, batch_size=None, num_workers=dataloader_workers)
    print(f'len(dataloader): {len(dataloader)}, len(dataset): {len(dataset)}, total_samples: {dataset.total_samples()}')
    t1 = time.time()
    h_div_w2samples = {}
    for ep in range(4):
        dataloader.dataset.set_epoch(ep)
        pbar = tqdm.tqdm(total=len(dataloader))
        for i, data in enumerate(iter(dataloader)):
            pbar.update(1)
            t2 = time.time()
            h_div_w = data[0].shape[-2] / data[0].shape[-1]
            h_div_w = f'{h_div_w:.3f}'
            if h_div_w not in h_div_w2samples:
                h_div_w2samples[h_div_w] = 0
            h_div_w2samples[h_div_w] += 1
            if (i+1) % 100 == 0:
                total_samples = np.sum(list(h_div_w2samples.values()))
                print()
                for h_div_w, num in sorted(h_div_w2samples.items()):
                    print(f'h_div_w: {h_div_w}, samples: {num}, proportion: {num/total_samples*100:.1f}%')
                print()
            t1 = time.time()
