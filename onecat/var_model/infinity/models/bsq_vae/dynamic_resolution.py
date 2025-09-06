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
full_ratio2hws = {}
for ratio, hws in ratio2hws.items():
    full_ratio2hws[ratio] = hws
    full_ratio2hws[int(1/ratio*1000)/1000] = [(item[1], item[0]) for item in hws]

dynamic_resolution_h_w = {}
predefined_HW_Scales_dynamic = {}
for ratio in full_ratio2hws:
    dynamic_resolution_h_w[ratio] ={}
    for ind, leng in enumerate([7, 10, 13]):
        h, w = full_ratio2hws[ratio][leng-1][0], full_ratio2hws[ratio][leng-1][1] # feature map size
        pixel = (h * vae_stride, w * vae_stride) # The original image (H, W)
        dynamic_resolution_h_w[ratio][pixel[1]] = {
            'pixel': pixel,
            'scales': full_ratio2hws[ratio][:leng]
        } # W as key
        predefined_HW_Scales_dynamic[(h, w)] = full_ratio2hws[ratio][:leng]