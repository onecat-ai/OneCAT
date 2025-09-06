# Overview
We provide [eval.sh](scripts/eval.sh) for evaluation on various benchmarks with only one command. In particular, [eval.sh](scripts/eval.sh) supports evaluation on commonly used metrics such as [GenEval](https://github.com/djghosh13/geneval), [ImageReward](https://github.com/THUDM/ImageReward), [HPSv2.1](https://github.com/tgxs002/HPSv2), FID and Validation Loss.

# Usage


## Basic Configuration

```shell
# set arguments
pn=1M
model_type=infinity_2b
infinity_model_path=[infinity_model_path]
out_dir_root=[out_dir_root]
vae_type=32
vae_path=[vae_path]
cfg=4
tau=1
text_encoder_ckpt=[text_encoder_ckpt]
text_channels=2048
sub_fix=cfg${cfg}_tau${tau}
```


## ImageReward
[ImageReward](https://github.com/THUDM/ImageReward) is a metric for evaluating the human preference score of generated images. It learns human preference through fine-tuning CLIP model with 137K human ranked image pairs.
```shell
out_dir=${out_dir_root}/image_reward_${sub_fix}
infer_eval_image_reward
```

## HPS v2.1
[HPSv2.1](https://github.com/tgxs002/HPSv2) is a metric for evaluating the human preference score of generated images. It learns human preference through fine-tuning CLIP model with 798K human ranked image pairs. The human ranked image pairs are from human experts.
```shell
out_dir=${out_dir_root}/hpsv21_${sub_fix}
infer_eval_hpsv21
```

## GenEval
[GenEval](https://github.com/djghosh13/geneval) is an object-focused framework for evaluating Text-to-Image alignment.
```shell
rewrite_prompt=0
out_dir=${out_dir_root}/gen_eval_${sub_fix}
test_gen_eval
```

## FID
For testing FID, you need provide a jsonl file which contains text prompts and ground truth images. We highly recommand the number of examples in the jsonl file is greater than 20000 since testing FID needs abundant of examples.
```shell
long_caption_fid=1
jsonl_filepath=[jsonl_filepath]
out_dir=${out_dir_root}/val_long_caption_fid_${sub_fix}
rm -rf ${out_dir}
test_fid
```

## Validation Loss
For testing Validation Loss, you need provide a jsonl folder like the training jsonl folder. Besides, you should specify the noise applying strength for Bitwise Self-Correction to the same strength used in the training phrase.
```shell
out_dir=${out_dir_root}/val_loss_${sub_fix}
reweight_loss_by_scale=0
jsonl_folder=[jsonl_folder]
noise_apply_strength=0.2
test_val_loss
```