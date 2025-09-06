import os
import numpy as np

import json


with open('/Users/bytedance/Desktop/projects/Infinity/evaluation/gen_eval/prompt_rewrite_cache_1.json', 'r') as f:
    correct = json.load(f)

with open('/Users/bytedance/Desktop/projects/Infinity/evaluation/gen_eval/prompt_rewrite_cache_123.json', 'r') as f:
    false_key_dict = json.load(f)

keys1_list = list(correct.keys())
keys2_list = list(false_key_dict.keys())

final_dict = {}
for i in range(len(keys1_list)):
    key1 = keys1_list[i]
    key2 = keys2_list[i]
    final_dict[key1] = false_key_dict[key2]

with open('prompt_rewrite_cache.json', 'w') as f:
    json.dump(final_dict, f, ensure_ascii=False, indent=2)
