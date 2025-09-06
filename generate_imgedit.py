import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from accelerate import Accelerator
from transformers import AutoTokenizer

from onecat.constants import *
from onecat.modeling_onecat import OneCatVLModel
from onecat.var_model.tools.run_infinity import load_visual_tokenizer
from onecat.smart_resize import smart_resize
from onecat.util import build_transform
from onecat.var_model.infinity.utils.dynamic_resolution import h_div_w_templates


def parse_args():
    parser = argparse.ArgumentParser(description="Generate edited images with OneCat model")

    # I/O
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--prompt", type=str, required=True, help="Edit instruction prompt.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to reference image.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or name of pretrained OneCat model.")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to Infinity VAE checkpoint.")

    # Generation controls
    parser.add_argument("--t2i_stage", type=int, default=3, help="T2I training stage indicator used for VAE config.")
    parser.add_argument("--cfg_I", type=float, default=1.0, help="Image CFG strength.")
    parser.add_argument("--cfg_T", type=float, default=3.0, help="Text CFG strength.")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.97, help="Top-p sampling.")
    parser.add_argument("--max_input_tokens", type=int, default=1024, help="Max tokens for tokenizer inputs.")

    args = parser.parse_args()
    return args


def center_crop_to_ratio(image: Image.Image, h_div_w: float) -> Image.Image:
    """
    Center crop the image to a target height/width ratio (h_div_w).
    Returns a new PIL.Image object.
    """
    width, height = image.size
    target_ratio = h_div_w

    if width * target_ratio > height:
        new_width = int(height / target_ratio)
        new_height = height
    else:
        new_width = width
        new_height = int(width * target_ratio)

    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    return image.crop((left, top, right, bottom))


def main(args):
    accelerator = Accelerator()
    device = accelerator.device

    #--------------------------------------------------------------------------
    # Create model and tokenizer.

    model = OneCatVLModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # VAE.
    vae_path = args.vae_path
    t2i_stage = args.t2i_stage
    if t2i_stage == 2:
        pn = '0.25M'
    elif t2i_stage == 3:
        pn = '1M'
    else:
        raise ValueError(f'Expected t2i_stage 2, or 3. Got {t2i_stage}')
    vae_args = argparse.Namespace(vae_type=32, vae_path=vae_path, apply_spatial_patchify=0, pn=pn)
    vae_local = load_visual_tokenizer(vae_args)
    vae_local.eval()

    # Setup model for generation.
    model.vae_local = vae_local
    model.vargpt_gen_args = vae_args
    img_gen_context_token_id = tokenizer.convert_tokens_to_ids(IMG_GEN_CONTEXT_TOKEN)
    img_gen_start_token_id = tokenizer.convert_tokens_to_ids(IMG_GEN_START_TOKEN)
    ref_img_context_token_id = tokenizer.convert_tokens_to_ids(REF_GEN_CONTEXT_TOKEN)
    model.img_gen_context_token_id = img_gen_context_token_id
    model.img_gen_start_token_id = img_gen_start_token_id
    model.ref_img_context_token_id = ref_img_context_token_id
    model.to(device=device, dtype=torch.bfloat16).eval()

    #--------------------------------------------------------------------------
    # Prepare input.

    prompt = args.prompt
    ref_image = Image.open(args.image_path).convert('RGB')
    
    # Process reference image
    h_div_w = ref_image.size[1] / ref_image.size[0]
    width, height = ref_image.size
    
    # Resize for processing
    if args.t2i_stage == 2:
        short_side_ref = 672
    elif args.t2i_stage == 3:
        short_side_ref = 1008
    else:
        raise ValueError(f'Expected t2i_stage 2 or 3. Got {args.t2i_stage}')
    
    h_div_w_template = h_div_w_templates[np.argmin(np.abs(float(h_div_w) - h_div_w_templates))]
    ref_image = center_crop_to_ratio(ref_image, h_div_w)
    width, height = ref_image.size
    
    if width < height:
        h_bar, w_bar = smart_resize(int(short_side_ref * h_div_w), short_side_ref)
    else:
        h_bar, w_bar = smart_resize(short_side_ref, int(short_side_ref / h_div_w))
    
    width, height = [w_bar, h_bar]
    ref_transform = build_transform(input_size=[height, width])
    
    # Calculate token dimensions
    patch_size = model.patch_size
    token_width = int(width // patch_size)
    token_height = int(height // patch_size)
    num_ref_image_token = int(token_width * token_height * (0.5 ** 2))
    
    ref_pixel_values = ref_transform(ref_image).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    
    # Prepare text input
    image_tokens = f'{REF_GEN_START_TOKEN}{REF_GEN_CONTEXT_TOKEN * num_ref_image_token}{REF_GEN_END_TOKEN}'
    prompt_with_image = image_tokens + prompt
    
    system_message = SYSTEM_PROMPT
    user_message = f'<|im_start|>user\n{prompt_with_image}<|im_end|>'
    assistant_message = '<|im_start|>assistant\n<img_gen>'
    batch = system_message + user_message + assistant_message
    
    model_inputs = tokenizer(batch, return_tensors='pt', max_length=args.max_input_tokens, truncation=False, padding=False)
    input_ids = model_inputs['input_ids'].to(device)
    attention_mask = model_inputs['attention_mask'].to(device)
    
    # CFG inputs
    cfg_batch = system_message + f'<|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant\n<img_gen>'
    model_inputs_cfg = tokenizer(cfg_batch, return_tensors='pt', max_length=args.max_input_tokens, truncation=False, padding=False)
    input_ids_cfg = model_inputs_cfg['input_ids'].to(device)
    attention_mask_cfg = model_inputs_cfg['attention_mask'].to(device)
    
    cfg_batch2 = system_message + f'<|im_start|>user<|im_end|><|im_start|>assistant\n<img_gen>'
    model_inputs_cfg2 = tokenizer(cfg_batch2, return_tensors='pt', max_length=args.max_input_tokens, truncation=False, padding=False)
    input_ids_cfg2 = model_inputs_cfg2['input_ids'].to(device)
    attention_mask_cfg2 = model_inputs_cfg2['attention_mask'].to(device)

    #--------------------------------------------------------------------------
    # Main generation loop.

    generation_config = dict(
        output_hidden_states=True,
        cfg_I=args.cfg_I,
        cfg_T=args.cfg_T,
        top_k=args.top_k,
        top_p=args.top_p,
        h_div_w=h_div_w_template,
        use_cache=True,
        return_dict=True,
    )
    with torch.no_grad():
        img = model.generate_edit(
            pixel_values=ref_pixel_values,
            input_ids=input_ids,
            input_ids_cfg=input_ids_cfg,
            input_ids_cfg2=input_ids_cfg2,
            attention_mask=attention_mask,
            attention_mask_cfg=attention_mask_cfg,
            attention_mask_cfg2=attention_mask_cfg2,
            **generation_config
        )
        img = Image.fromarray(img[0].add_(1).mul_(0.5).to(dtype=torch.float32).permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8))
    
    # Save images.
    folder_path = Path(args.output_dir)
    folder_path.mkdir(parents=True, exist_ok=True)
    img.save(folder_path / "edited.png")
    print(f"Generated edited image. Results are saved to {folder_path}")

    #--------------------------------------------------------------------------


if __name__ == "__main__":
    args = parse_args()
    main(args)