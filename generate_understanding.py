import argparse
import torch
from PIL import Image

from transformers import AutoTokenizer
from accelerate import Accelerator

from onecat.util import build_transform
from onecat.smart_resize import smart_resize
from onecat.modeling_onecat import OneCatVLModel


def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    width, height = image.size
    
    resized_height, resized_width = smart_resize(height,width) 
    transform = build_transform(input_size=(resized_height, resized_width))
    images = [image]
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    
    transform_base = build_transform(input_size=(448, 448))
    pixel_values_thumbnail = [transform_base(image) for image in images]
    pixel_values_thumbnail = torch.stack(pixel_values_thumbnail)
    
    return pixel_values, pixel_values_thumbnail


def main(args):
    accelerator = Accelerator()
    device = accelerator.device

    # Prepare model.
    model = OneCatVLModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Prepare input.
    question = args.question
    img_path = args.img_path
    pixel_values, pixel_values_thumbnail = load_image(img_path)
    pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
    pixel_values_thumbnail = pixel_values_thumbnail.to(device=device, dtype=torch.bfloat16)
    
    # Generate.
    model.to(device=device, dtype=torch.bfloat16).eval()
    generation_config = dict(
        do_sample=False,
        top_k=None,
        top_p=None,
        num_beams=1,
        max_new_tokens=1000,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question=question,
        generation_config=generation_config,
        pixel_values_thumbnail=pixel_values_thumbnail,
        verbose=False
    )

    print()
    print(response)
    print()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path")
    p.add_argument("--img_path")
    p.add_argument("--question")
    args = p.parse_args()
    main(args)
