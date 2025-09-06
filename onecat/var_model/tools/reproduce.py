import random
import torch
import os
import os.path as osp
import cv2
import numpy as np
from run_infinity import *

torch.cuda.set_device(0)
model_path = '/workspace/Infinity/weights/infinity_2b_reg.pth'
vae_path = '/workspace/Infinity/weights/infinity_vae_d32reg.pth'
text_encoder_ckpt = '/workspace/Infinity/weights/flan-t5-xl'

# SET
args = argparse.Namespace(
    pn='1M',
    model_path=model_path,
    cfg_insertion_layer=0,
    vae_type=32,
    vae_path=vae_path,
    add_lvl_embeding_only_first_block=1,
    use_bit_label=1,
    model_type='infinity_2b',
    rope2d_each_sa_layer=1,
    rope2d_normalized_by_hw=2,
    use_scale_schedule_embedding=0,
    sampling_per_bits=1,
    text_encoder_ckpt=text_encoder_ckpt,
    text_channels=2048,
    apply_spatial_patchify=0,
    h_div_w_template=1.000,
    use_flex_attn=0,
    cache_dir='/dev/shm',
    checkpoint_type='torch',
    seed=0,
    bf16=1,
    save_file='tmp.jpg',
    enable_model_cache=0
)

# LOAD
text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
vae = load_visual_tokenizer(args)
infinity = load_transformer(vae, args)

# PROMPT
prompts = {
    "vintage_insect": "Insect made from vintage 1960s electronic components, capacitors, resistors, transistors, wires, diodes, solder, circuitboard.",
    "macro_closeup": "Denis Villeneuve's extreme macro cinematographic close-up in water.",
    "3d_school": "A creative 3D image to be placed at the bottom of a mobile application's homepage, depicting a miniature school and children carrying backpacks.",
    "explore_more": "Create an image with 'Explore More' in an adventurous font over a picturesque hiking trail.",
    "toy_car": "Close-up shot of a diecast toy car, diorama, night, lights from windows, bokeh, snow.",
    "fairy_house": "House: white; pink tinted windows; surrounded by flowers; cute; scenic; garden; fairy-like; epic; photography; photorealistic; insanely detailed and intricate; textures; grain; ultra-realistic.",
    "cat_fashion": "Hyperrealistic black and white photography of cats fashion show in style of Helmut Newton.",
    "spacefrog_astroduck": "Two superheroes called Spacefrog (a dashing green cartoon-like frog with a red cape) and Astroduck (a yellow fuzzy duck, part-robot, with blue/grey armor), near a garden pond, next to their spaceship, a classic flying saucer, called the Tadpole 3000. Photorealistic.",
    "miniature_village": "An enchanted miniature village bustling with activity, featuring tiny houses, markets, and residents.",
    "corgi_dog": "A close-up photograph of a Corgi dog. The dog is wearing a black hat and round, dark sunglasses. The Corgi has a joyful expression, with its mouth open and tongue sticking out, giving an impression of happiness or excitement.",
    "robot_eggplant": "a robot holding a huge eggplant, sunny nature background",
    "perfume_product": "Product photography, a perfume placed on a white marble table with pineapple, coconut, lime next to it as decoration, white curtains, full of intricate details, realistic, minimalist, layered gestures in a bright and concise atmosphere, minimalist style.",
    "mountain_landscape": "The image presents a picturesque mountainous landscape under a cloudy sky. The mountains, blanketed in lush greenery, rise majestically, their slopes dotted with clusters of trees and shrubs. The sky above is a canvas of blue, adorned with fluffy white clouds that add a sense of tranquility to the scene. In the foreground, a valley unfolds, nestled between the towering mountains. It appears to be a rural area, with a few buildings and structures visible, suggesting the presence of a small settlement. The buildings are scattered, blending harmoniously with the natural surroundings. The image is captured from a high vantage point, providing a sweeping view of the valley and the mountains."
}

# OUTPUT
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# GEN IMG
for category, prompt in prompts.items():
    cfg = 3
    tau = 0.5
    h_div_w = 1/1 # Aspect Ratio
    seed = random.randint(0, 10000)
    enable_positive_prompt = 0

    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    # GEN
    generated_image = gen_one_img(
        infinity,
        vae,
        text_tokenizer,
        text_encoder,
        prompt,
        g_seed=seed,
        gt_leak=0,
        gt_ls_Bl=None,
        cfg_list=cfg,
        tau_list=tau,
        scale_schedule=scale_schedule,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type,
        sampling_per_bits=args.sampling_per_bits,
        enable_positive_prompt=enable_positive_prompt,
    )

    # SAVE
    save_path = osp.join(output_dir, f"re_{category}_test.jpg")
    cv2.imwrite(save_path, generated_image.cpu().numpy())
    print(f"{category} image saved to {save_path}")
