import json

import cv2
import torch
# import traceback
torch._dynamo.config.cache_size_limit = 64

from run_infinity import *


def process_short_text(short_text):
    if '--' in short_text:
        processed_text = short_text.split('--')[0]
        if processed_text:
            short_text = processed_text
    return short_text

np.random.seed(0)

lines4infer = []
prompt_list = [
    'A high-contrast photo of a panda riding a horse. The panda is wearing a wizard hat and is reading a book. The horse is standing on a street against a gray concrete wall. Colorful flowers and the word "PEACE" are painted on the wall. Green grass grows from cracks in the street. DSLR photograph. daytime lighting.',
    'a red cube on the top of blue sphere, the behind is a yellow triangle. A cat is on the left and a dog is on the right',
    'The red hat was on the left of the blue backpack.',
    'woman lying on the beach',
    'woman laying on grass',
    'Epic anime artwork of a wizard atop a mountain at night casting a cosmic spell into the dark sky that says "VAR" made out of colorful energy',
    'A board with text "Hello, VAR"',
    'A paper reads "No!"',
    'A photograph featuring a young woman standing in a field of tall green plants, possibly corn, with the sun shining through the foliage creating a warm, golden glow. The woman is looking off to the side with a gentle expression, and her face is partially obscured by the plants. The sunlight creates a lens flare effect, adding a dreamy quality to the image. The style of the image is naturalistic, capturing a moment of serenity in a rural setting.',
    "A photo-realistic picture. A black and white photograph that captures a man in profile. The man has a beard and mustache, and his hair appears to be swept back. He is wearing a scarf or shawl that is wrapped around his neck and shoulders, adding texture to the image. The photograph is taken from a close-up angle, focusing on the man's face and the upper part of his torso. The lighting is dramatic, with strong contrasts between light and shadow, highlighting the contours of his face and the texture of his hair and clothing. The style of the image is reminiscent of a cinematic or artistic portrait, emphasizing mood and emotion over realism.",
    'A man engaged in the activity of paddleboarding. He is balancing on a white paddleboard with a pink nose, which is partially submerged in the blue water. The man is wearing a black sleeveless top, blue shorts, and sunglasses. His hair is long and appears to be wet, suggesting he has been in the water. He is smiling and seems to be enjoying the moment, with his arms outstretched for balance. The background shows a clear sky and distant mountains, indicating that the setting is likely a large body of water, such as a lake or sea, on a sunny day. The photograph is taken in a realistic style, capturing the action and the natural environment.',
    'a young woman standing in the grass,, in the style of stark black-and-white photography,, hasselblad 1600f,, coastal landscapes,, expressive facial features,, dutch landscapes,, soft atmospheric scenes,, powerful portraits',
    'a digital painting of an old man with a beard and some dark grays,, in the style of photorealistic urban scenes,, uhd image,, algeapunk,, rusty debris,, vibrant portraits,, flickr,, soft-focus portraits',
    'beautiful female warrior,, short blue hair,, shimmering jewels armor,, in the style of Alfons Mucha,, with emphasis on light play and the transparency of the glass,, High and short depth of field,, Ray tracing,, FHD,, hyper quality',
    'a young female hobbit,, ultra realism,, lord of the rings,, snowy forest,, pale hues,, hobbit from lord of the rings who escaped Carn Dum,, grimy,, dirty,, black hair,, homely,, ugly',
    'A dog is walking on a leash with its owner.',
    'A man is running a marathon and crossing the finish line.',
    'an oblong eggplant and a teardrop pear',
    'an oblong cucumber and a teardrop pepper',
    'a brown dog and a blue horse',
    'a rabbit fights with a tiger',
    'three women',
    'three deer',
    'a tree',
    'a photo of a tree',
    'grassland',
    'a woman rides a tiger in the forest',
    'a beautiful scenery area of russia',
    'an oil painting of a house',
    "two girls",
    "three boys",
    'two candles on a marble table next to a silver spoon',
    'woman lying on the beach',
    'woman laying on grass',
    'woman laying on the beach',
    'liberty of statue',
    'a man full body shot',
    'a woman full body shot',
    'a set of sushi which consists of a US map shape',
    'Asian girl near the beach',
    'two women sitting in the sofa and hold red wine cup',
    'a rabbit fights with a tiger',
    'two ninjas fight with each other during night',
    'a red cube on the top of blue sphere, the behind is a yellow triangle. A cat is on the left and a dog is on the right',
    'Epic anime artwork of a wizard atop a mountain at night casting a cosmic spell into the dark sky that says "VAR" made out of colorful energy',
    'a woman having a spa day',
    'two men boxing',
    'a Chinese woman laying on the beach',
    'a man laying on a bed',
    'A brand with "VAR DIT"',
    'A board with text "Hello, VAR"',
    'A paper reads "No!"',
    'A paper reads "VAR Yes!"',
    'American national flag',
    'China national flag',
    'Russia national flag',
    'a woman lying on the beach sunbathing',
    'ironman',
    "Generate the text 'happy' with autumn leaves and cold colors.",
    "Generate the text 'bytedance' with autumn leaves and cold colors.",
    "Generate the text 'GenAI' in a supermarket.",
    "Generate the text 'GenAI' in a grass.",
    "Generate the text 'GenAI' in a ground.",
    "Generate the text 'KCTG' in a book.",
    "Generate the text 'GenAI' in a table.",
    "a Chinese model is sitting on a train, magazine cover, photorealistic, futuristic style",
]

for prompt in prompt_list:
    lines4infer.append({
        'prompt': prompt,
        'h_div_w': 1.0,
        'infer_type': 'infer/free_prompt',
    })

candidates = []
for candidate in candidates:
    jsonl_filepath = candidate['jsonl_filepath']
    if not osp.exists(jsonl_filepath):
        continue
    with open(jsonl_filepath, 'r') as f:
        cnt = 0
        for line in f:
            meta = json.loads(line)
            gt_image_path = meta['image_path']
            assert osp.exists(gt_image_path), gt_image_path
            if meta['text'] and (meta['long_caption'] != meta['text']):
                lines4infer.append({
                    'prompt': meta['text'],
                    'h_div_w': meta['h_div_w'],
                    'infer_type': candidate['infer_type'],
                    'gt_image_path': gt_image_path,
                })
            lines4infer.append({
                'prompt': meta['long_caption'],
                'h_div_w': meta['h_div_w'],
                'infer_type': candidate['infer_type'],
                'gt_image_path': gt_image_path,
            })
            cnt += 1
            if cnt > candidate['sample_num']:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument('--rewrite_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--coco30k_prompts', type=int, default=0, choices=[0,1])
    parser.add_argument('--save4fid_eval', type=int, default=0, choices=[0,1])
    parser.add_argument('--save_recons_img', type=int, default=0, choices=[0,1])
    parser.add_argument('--jsonl_filepath', type=str, default='')
    parser.add_argument('--long_caption_fid', type=int, default=1, choices=[0,1])
    parser.add_argument('--fid_max_examples', type=int, default=-1)
    parser.add_argument('--n_samples', type=int, default=1)
    args = parser.parse_args()
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
        
    if args.coco30k_prompts:
        from T2IBenchmark.datasets import get_coco_30k_captions, get_coco_fid_stats
        id2caption = get_coco_30k_captions()
        captions = []
        ids = []
        lines4infer = []
        for d in id2caption.items():
            ids.append(d[0])
            captions.append(d[1])
        np.random.shuffle(captions)
        lines4infer = [{'prompt': prompt, 'h_div_w': 1.0, 'infer_type': 'infer/coco30k_prompt'} for prompt in captions]
    
    if args.jsonl_filepath:
        lines4infer = []
        with open(args.jsonl_filepath, 'r') as f:
            cnt = 0
            for line in f:
                meta = json.loads(line)
                gt_image_path = meta['image_path']
                assert osp.exists(gt_image_path), gt_image_path
                if args.long_caption_fid:
                    prompt = meta['long_caption']
                else:
                    prompt = meta['text']
                if not prompt:
                    continue
                lines4infer.append({
                    'prompt': prompt,
                    'h_div_w': meta['h_div_w'],
                    'infer_type': 'val/laion_coco_long_caption',
                    'gt_image_path': gt_image_path,
                    'meta_line': line,
                })

    if args.fid_max_examples > 0:
        lines4infer = lines4infer[:args.fid_max_examples]
    print(f'Totally {len(lines4infer)} items for infer')

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = osp.join('output', osp.basename(osp.dirname(model_path)), osp.splitext(osp.basename(model_path))[0], f'coco30k_infer' if args.coco30k_prompts else 'comprehensive_infer')
    print(f'save to {out_dir}')

    # load text encoder
    text_tokenizer, text_encoder = load_tokenizer(t5_path =args.text_encoder_ckpt)
    # load vae
    vae = load_visual_tokenizer(args)
    # load infinity
    infinity = load_transformer(vae, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # inference
    if osp.exists(out_dir):
        # shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(__file__, osp.join(out_dir, osp.basename(__file__)))

    jsonl_list = []
    cnt = 0
    for i, infer_data in enumerate(lines4infer):
        try:
            prompt = infer_data['prompt']
            prompt = process_short_text(prompt)
            prompt_id = get_prompt_id(prompt)
            save_file = osp.join(out_dir, 'pred', f'{prompt_id}.jpg')
            if osp.exists(save_file):
                continue

            h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - infer_data['h_div_w']))]
            scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
            scale_schedule = [(1, h, w) for (t,h, w) in scale_schedule]
            if args.apply_spatial_patchify:
                vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
            else:
                vae_scale_schedule = scale_schedule
            tgt_h, tgt_w = dynamic_resolution_h_w[h_div_w_template][args.pn]['pixel']

            img_list = []
            gt_ls_Bl = []
            if ('gt_image_path' in infer_data):
                gt_img, recons_img, all_bit_indices = joint_vi_vae_encode_decode(vae, infer_data['gt_image_path'], vae_scale_schedule, device, tgt_h, tgt_w)
                gt_ls_Bl = all_bit_indices
            else:
                if args.save4fid_eval:
                    continue
            
            if args.coco30k_prompts or args.save4fid_eval:
                concate_img = gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, g_seed=0, gt_leak=0, gt_ls_Bl=gt_ls_Bl, t5_path=None, tau_list=args.tau, cfg_sc=3, cfg_list=args.cfg, scale_schedule=scale_schedule, cfg_insertion_layer=[args.cfg_insertion_layer], vae_type=args.vae_type, sampling_per_bits=args.sampling_per_bits)
            else:
                g_seed = 0 if args.n_samples == 1 else None
                tmp_img_list = []
                for _ in range(args.n_samples):
                    tmp_img_list.append(gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, g_seed=g_seed, gt_leak=0, gt_ls_Bl=gt_ls_Bl, t5_path=None, tau_list=args.tau, cfg_sc=3, cfg_list=args.cfg, scale_schedule=scale_schedule, cfg_insertion_layer=[args.cfg_insertion_layer], vae_type=args.vae_type, sampling_per_bits=1, top_k=0))
                img_list.append(np.concatenate(tmp_img_list, axis=1))

                tmp_img_list = []
                for _ in range(args.n_samples):
                    tmp_img_list.append(gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, g_seed=g_seed, gt_leak=0, gt_ls_Bl=gt_ls_Bl, t5_path=None, tau_list=args.tau, cfg_sc=3, cfg_list=args.cfg, scale_schedule=scale_schedule, cfg_insertion_layer=[args.cfg_insertion_layer], vae_type=args.vae_type, sampling_per_bits=8, top_k=0))
                img_list.append(np.concatenate(tmp_img_list, axis=1))

                tmp_img_list = []
                for _ in range(args.n_samples):
                    tmp_img_list.append(gen_one_img(infinity, vae, text_tokenizer, text_encoder, prompt, g_seed=g_seed, gt_leak=0, gt_ls_Bl=gt_ls_Bl, t5_path=None, tau_list=args.tau, cfg_sc=3, cfg_list=args.cfg, scale_schedule=scale_schedule, cfg_insertion_layer=[args.cfg_insertion_layer], vae_type=args.vae_type, sampling_per_bits=16, top_k=0))
                img_list.append(np.concatenate(tmp_img_list, axis=1))
                
                if args.n_samples == 1:
                    concate_img = np.concatenate([np.array(item) for item in img_list], 1)
                else:
                    concate_img = np.concatenate([np.array(item) for item in img_list], 0)
                concate_img = Image.fromarray(concate_img)
            
            os.makedirs(osp.dirname(save_file), exist_ok=True)
            cv2.imwrite(save_file, concate_img.cpu().numpy())
            infer_data['image_path'] = osp.abspath(save_file)

            if args.save4fid_eval:
                save_file = osp.join(out_dir, 'gt', f'{prompt_id}.jpg')
                os.makedirs(osp.dirname(save_file), exist_ok=True)
                gt_img = Image.fromarray(np.array(gt_img))
                assert not osp.exists(save_file), f'{save_file} exists, infer_data: {infer_data}'
                gt_img.save(save_file)
                if args.save_recons_img:
                    save_file = osp.join(out_dir, 'recons', f'{prompt_id}.jpg')
                    os.makedirs(osp.dirname(save_file), exist_ok=True)
                    recons_img = Image.fromarray(np.array(recons_img))
                    recons_img.save(save_file)

            jsonl_list.append(json.dumps(infer_data)+'\n')
            jsonl_file = osp.join(out_dir, 'meta_info.jsonl')
            with open(jsonl_file, 'w') as f:
                f.writelines(jsonl_list)
            print(f'Save to {osp.abspath(jsonl_file)}')
        except Exception as e:
            # print(f"{e}", traceback.print_exc())
            print(f"{e}")

