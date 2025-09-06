import os
import os.path as osp
import time
import itertools
import shutil
import glob
import argparse

import tqdm
import numpy as np
import threading

def save_lines(lines, filename):
    os.makedirs(osp.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.writelines(lines)
    del lines

def get_part_jsonls(filepath, total_line_number, parts=512):
    dirname, filename, ext = osp.dirname(filepath), osp.splitext(osp.basename(filepath))[0], osp.splitext(osp.basename(filepath))[1]
    if parts == 1:
        return False, {1: filepath}
    save_dir = osp.join(dirname, f'{parts:04d}_parts')
    chunk_id2save_files = {}
    missing = False
    chunk_size = int(total_line_number/parts)
    for chunk_id in range(1, parts+1):
        if chunk_id == parts:
            num_of_lines = total_line_number - chunk_size * (parts-1)
        else:
            num_of_lines = chunk_size
        chunk_id2save_files[chunk_id] = osp.join(save_dir, f'{filename}_{chunk_id:04d}_{parts:04d}_{num_of_lines:09d}{ext}')
        if not osp.exists(chunk_id2save_files[chunk_id]):
            missing = True
    return missing, chunk_id2save_files

def split_large_txt_files(filepath, chunk_id2save_files):
    thread_list = []
    chunk_id = 1
    with open(filepath, 'r') as f:
        chunk = []
        pbar = tqdm.tqdm(total=len(chunk_id2save_files))
        for line in f:
            chunk.append(line)
            cur_chunk_size = int(osp.splitext(osp.basename(chunk_id2save_files[chunk_id]))[0].split('_')[-1])
            if len(chunk) >= cur_chunk_size:
                pbar.update(1)
                thread_list.append(threading.Thread(target=save_lines, args=(chunk, chunk_id2save_files[chunk_id])))
                thread_list[-1].start()
                chunk = []
                chunk_id += 1
        if len(chunk):
            import ipdb; ipdb.set_trace()
        assert not len(chunk)
        for thread in thread_list:
            thread.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl_folder', type=str, default='')
    parser.add_argument('--parts', type=int, default=600)
    args = parser.parse_args()
    for jsonl_filepath in sorted(glob.glob(osp.join(args.jsonl_folder, '*.jsonl'))):
        print(jsonl_filepath)
        t1 = time.time()
        line_num = int(jsonl_filepath.split('_')[-1].split('.')[0])
        missing, chunk_id2save_files = get_part_jsonls(jsonl_filepath, line_num, parts=args.parts)
        split_large_txt_files(jsonl_filepath, chunk_id2save_files)
        t2 = time.time()
        print(f'split takes {t2-t1}s')
