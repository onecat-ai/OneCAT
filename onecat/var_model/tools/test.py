
from run_infinity import *
model_path='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/lihan/mmpretrain/huggingface.co/FoundationVision/Infinity/infinity_2b_reg.pth'
vae_path='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/lihan/mmpretrain/huggingface.co/FoundationVision/Infinity/infinity_vae_d32reg.pth'
     

args=argparse.Namespace(
pn='0.06M',
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
text_encoder_ckpt="",
text_channels=2048,
apply_spatial_patchify=0,
h_div_w_template=1.000,
use_flex_attn=0,
cache_dir='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/lihan/.cache',
checkpoint_type='torch',
seed=0,
bf16=1,
save_file='tmp.jpg',
enable_model_cache=False,
bitloss_type='mean',
reweight_loss_by_scale=1,
)
# load v

vae_local = load_visual_tokenizer(args).cuda()
# vae_local_weight = torch.load('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/lihan/mmpretrain/huggingface.co/FoundationVision/Infinity/infinity_vae_d32reg.pth')
print(vae_local.encoder.conv_in.conv.weight)