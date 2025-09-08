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


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with OneCat model")

    # I/O
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--prompt", type=str, required=True, help="Path to a text file with one prompt per line.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or name of pretrained OneCat model.")
    parser.add_argument("--vae_path", type=str, required=True, help="Path to Infinity VAE checkpoint.")

    # Generation controls
    parser.add_argument("--t2i_stage", type=int, default=3, help="T2I training stage indicator used for VAE config.")
    parser.add_argument("--h_div_w", type=float, default=1.0, help="Height divided by width of generated images.")
    parser.add_argument("--num_imgs_per_text", type=int, default=1, help="Number of images to generate per prompt.")
    parser.add_argument("--cfg", type=float, default=5, help="Classifier-free guidance strength.")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k sampling.")
    parser.add_argument("--top_p", type=float, default=0.97, help="Top-p sampling.")
    parser.add_argument("--max_input_tokens", type=int, default=1024, help="Max tokens for tokenizer inputs.")

    args = parser.parse_args()
    return args


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
        raise ValueError(f'Expected t2i_stage 2 or 3. Got {t2i_stage}')
    vae_args = argparse.Namespace(vae_type=32, vae_path=vae_path, apply_spatial_patchify=0, pn=pn)
    vae_local = load_visual_tokenizer(vae_args)
    vae_local.eval()

    # Setup model for generation.
    model.vae_local = vae_local
    model.vargpt_gen_args = vae_args
    img_gen_context_token_id = tokenizer.convert_tokens_to_ids(IMG_GEN_CONTEXT_TOKEN)
    img_gen_start_token_id = tokenizer.convert_tokens_to_ids(IMG_GEN_START_TOKEN)
    model.img_gen_context_token_id = img_gen_context_token_id
    model.img_gen_start_token_id = img_gen_start_token_id
    model.to(device=device, dtype=torch.bfloat16).eval()

    #--------------------------------------------------------------------------
    # Prepare input.

    prompt = args.prompt
    system_message = SYSTEM_PROMPT
    user_message = f'<|im_start|>user\n{prompt}<|im_end|>'
    assistant_message = '<|im_start|>assistant\n<img_gen>'
    
    batch = system_message + user_message + assistant_message
    model_inputs = tokenizer(batch, return_tensors='pt', max_length=args.max_input_tokens, truncation=False, padding=False)
    input_ids = model_inputs['input_ids'].to(device)
    attention_mask = model_inputs['attention_mask'].to(device)

    cfg_batch = system_message + '<|im_start|>user\n<|im_end|><|im_start|>assistant\n<img_gen>'
    model_inputs_cfg = tokenizer(cfg_batch, return_tensors='pt', max_length=args.max_input_tokens, truncation=False, padding=False)
    input_ids_cfg = model_inputs_cfg['input_ids'].to(device)
    attention_mask_cfg = model_inputs_cfg['attention_mask'].to(device)

    #--------------------------------------------------------------------------
    # Main generation loop.

    generation_config = dict(
        output_hidden_states=True,
        cfg=args.cfg,
        top_k=args.top_k,
        top_p=args.top_p,
        use_cache=True,
        return_dict=True,
        h_div_w=args.h_div_w,
    )
    imgs = []
    with torch.no_grad():
        for _ in range(int(args.num_imgs_per_text)):
            img = model.generate_t2i(
                input_ids=input_ids,
                input_ids_cfg=input_ids_cfg,
                attention_mask=attention_mask,
                attention_mask_cfg=attention_mask_cfg,
                **generation_config
            )            
            img = Image.fromarray(img[0].add_(1).mul_(0.5).to(dtype=torch.float32).permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8))
            imgs.append(img)
        
    # Save images.
    folder_path = Path(args.output_dir)
    folder_path.mkdir(parents=True, exist_ok=True)
    for j, im in enumerate(imgs):
        im.save(folder_path / f"{j:05d}.png")
    print(f"Generated {args.num_imgs_per_text} images. Results are saved to {folder_path}")

    #--------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    main(args)