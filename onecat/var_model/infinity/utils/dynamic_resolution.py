import json
import numpy as np
import tqdm

vae_stride = 16
ratio2hws = {
    1.000: [(1,1),(2,2),(4,4),(6,6),(8,8),(12,12),(16,16),(20,20),(24,24),(32,32),(40,40),(48,48),(64,64)],
    1.250: [(1,1),(2,2),(3,3),(5,4),(10,8),(15,12),(20,16),(25,20),(30,24),(35,28),(45,36),(55,44),(70,56)],
    1.333: [(1,1),(2,2),(4,3),(8,6),(12,9),(16,12),(20,15),(24,18),(28,21),(36,27),(48,36),(60,45),(72,54)],
    1.500: [(1,1),(2,2),(3,2),(6,4),(9,6),(15,10),(21,14),(27,18),(33,22),(39,26),(48,32),(63,42),(78,52)],
    1.750: [(1,1),(2,2),(3,3),(7,4),(11,6),(14,8),(21,12),(28,16),(35,20),(42,24),(56,32),(70,40),(84,48)],
    2.000: [(1,1),(2,2),(4,2),(6,3),(10,5),(16,8),(22,11),(30,15),(38,19),(46,23),(60,30),(74,37),(90,45)],
    2.500: [(1,1),(2,2),(5,2),(10,4),(15,6),(20,8),(25,10),(30,12),(40,16),(50,20),(65,26),(80,32),(100,40)],
    3.000: [(1,1),(2,2),(6,2),(9,3),(15,5),(21,7),(27,9),(36,12),(45,15),(54,18),(72,24),(90,30),(111,37)],
}
predefined_t = [1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15, 17, 21]

full_ratio2hws = {}
for ratio, hws in ratio2hws.items():
    full_ratio2hws[ratio] = hws
    if ratio != 1.000:
        full_ratio2hws[int(1/ratio*1000)/1000] = [(item[1], item[0]) for item in hws]

dynamic_resolution_h_w = {}
for ratio in full_ratio2hws:
    dynamic_resolution_h_w[ratio] ={}
    for ind, leng in enumerate([7, 10, 12, 13]):
        h_div_w = full_ratio2hws[ratio][leng-1][0] / full_ratio2hws[ratio][leng-1][1]
        assert np.abs(h_div_w-ratio) < 0.01, f'{full_ratio2hws[ratio][leng-1]}: {h_div_w} != {ratio}'
        pixel = (full_ratio2hws[ratio][leng-1][0] * vae_stride, full_ratio2hws[ratio][leng-1][1] * vae_stride)
        if ind == 0:
            total_pixels = '0.06M'
        elif ind == 1:
            total_pixels = '0.25M'
        elif ind == 2:
            total_pixels = '0.60M'
        else:
            total_pixels = '1M'
        
        scales = full_ratio2hws[ratio][:leng]
        scales = [ (t, h, w) for t, (h, w) in zip(predefined_t, scales) ]
        dynamic_resolution_h_w[ratio][total_pixels] = {
            'pixel': pixel,
            'scales': scales
        }

h_div_w_templates = []
for h_div_w in dynamic_resolution_h_w.keys():
    h_div_w_templates.append(h_div_w)
h_div_w_templates = np.array(h_div_w_templates)

def get_h_div_w_template2indices(h_div_w_list, h_div_w_templates):
    indices = list(range(len(h_div_w_list)))
    h_div_w_template2indices = {}
    pbar = tqdm.tqdm(total=len(indices), desc='get_h_div_w_template2indices...')
    for h_div_w, index in zip(h_div_w_list, indices):
        pbar.update(1)
        nearest_h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
        if nearest_h_div_w_template_ not in h_div_w_template2indices:
            h_div_w_template2indices[nearest_h_div_w_template_] = []
        h_div_w_template2indices[nearest_h_div_w_template_].append(index)
    for h_div_w_template_, sub_indices in h_div_w_template2indices.items():
        h_div_w_template2indices[h_div_w_template_] = np.array(sub_indices)
    return h_div_w_template2indices

if __name__ == '__main__':
    for h_div_w_template in dynamic_resolution_h_w:
        for total_pixels in dynamic_resolution_h_w[h_div_w_template]:
            scales = np.array(dynamic_resolution_h_w[h_div_w_template][total_pixels]['scales'])
            seq_len = np.sum(scales[:,0]*scales[:,1])
            if total_pixels == '1M':
                string = f'{h_div_w_template}, {total_pixels}, {dynamic_resolution_h_w[h_div_w_template][total_pixels]}, seq_len: {seq_len}'.replace(', ', ',')
                print(string)
