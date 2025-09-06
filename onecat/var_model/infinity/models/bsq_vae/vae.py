import argparse
import torch

from onecat.var_model.infinity.models.bsq_vae.flux_vqgan import AutoEncoder

def load_cnn(model, state_dict, prefix, expand=False, use_linear=False):
    delete_keys = []
    loaded_keys = []
    for key in state_dict:
        if key.startswith(prefix):
            _key = key[len(prefix):]
            if _key in model.state_dict():
                # load nn.Conv2d or nn.Linear to nn.Linear
                if use_linear and (".q.weight" in key or ".k.weight" in key or ".v.weight" in key or ".proj_out.weight" in key):
                    load_weights = state_dict[key].squeeze()
                elif _key.endswith(".conv.weight") and expand:
                    if model.state_dict()[_key].shape == state_dict[key].shape:
                        # 2D cnn to 2D cnn
                        load_weights = state_dict[key]
                    else:
                        # 2D cnn to 3D cnn
                        _expand_dim = model.state_dict()[_key].shape[2]
                        load_weights = state_dict[key].unsqueeze(2).repeat(1, 1, _expand_dim, 1, 1)
                else:
                    load_weights = state_dict[key]
                model.state_dict()[_key].copy_(load_weights)
                delete_keys.append(key)
                loaded_keys.append(prefix+_key)
            # load nn.Conv2d to Conv class
            conv_list = ["conv"] if use_linear else ["conv", ".q.", ".k.", ".v.", ".proj_out.", ".nin_shortcut."]
            if any(k in _key for k in conv_list):
                if _key.endswith(".weight"):
                    conv_key = _key.replace(".weight", ".conv.weight")
                    if conv_key and conv_key in model.state_dict():
                        if model.state_dict()[conv_key].shape == state_dict[key].shape:
                            # 2D cnn to 2D cnn
                            load_weights = state_dict[key]
                        else:
                            # 2D cnn to 3D cnn
                            _expand_dim = model.state_dict()[conv_key].shape[2]
                            load_weights = state_dict[key].unsqueeze(2).repeat(1, 1, _expand_dim, 1, 1)
                        model.state_dict()[conv_key].copy_(load_weights)
                        delete_keys.append(key)
                        loaded_keys.append(prefix+conv_key)
                if _key.endswith(".bias"):
                    conv_key = _key.replace(".bias", ".conv.bias")
                    if conv_key and conv_key in model.state_dict():
                        model.state_dict()[conv_key].copy_(state_dict[key])
                        delete_keys.append(key)
                        loaded_keys.append(prefix+conv_key)
            # load nn.GroupNorm to Normalize class
            if "norm" in _key:
                if _key.endswith(".weight"):
                    norm_key = _key.replace(".weight", ".norm.weight")
                    if norm_key and norm_key in model.state_dict():
                        model.state_dict()[norm_key].copy_(state_dict[key])
                        delete_keys.append(key)
                        loaded_keys.append(prefix+norm_key)
                if _key.endswith(".bias"):
                    norm_key = _key.replace(".bias", ".norm.bias")
                    if norm_key and norm_key in model.state_dict():
                        model.state_dict()[norm_key].copy_(state_dict[key])
                        delete_keys.append(key)
                        loaded_keys.append(prefix+norm_key)
            
    for key in delete_keys:
        del state_dict[key]

    return model, state_dict, loaded_keys


def vae_model(vqgan_ckpt, schedule_mode, codebook_dim, codebook_size, test_mode=True, patch_size=16, encoder_ch_mult=[1, 2, 4, 4, 4], decoder_ch_mult=[1, 2, 4, 4, 4],):
    args=argparse.Namespace(
        vqgan_ckpt=vqgan_ckpt,
        sd_ckpt=None,
        inference_type='image',
        save='./imagenet_val_bsq',
        save_prediction=True,
        image_recon4video=False,
        junke_old=False,
        device='cuda',
        max_steps=1000000.0,
        log_every=1,
        visu_every=1000,
        ckpt_every=1000,
        default_root_dir='',
        compile='no',
        ema='no',
        lr=0.0001,
        beta1=0.9,
        beta2=0.95,
        warmup_steps=0,
        optim_type='Adam',
        disc_optim_type=None,
        lr_min=0.0,
        warmup_lr_init=0.0,
        max_grad_norm=1.0,
        max_grad_norm_disc=1.0,
        disable_sch=False,
        patch_size=patch_size,
        temporal_patch_size=4,
        embedding_dim=256,
        codebook_dim=codebook_dim,
        num_quantizers=8,
        quantizer_type='MultiScaleBSQ',
        use_vae=False,
        use_freq_enc=False,
        use_freq_dec=False,
        preserve_norm=False,
        ln_before_quant=False,
        ln_init_by_sqrt=False,
        use_pxsf=False,
        new_quant=True,
        use_decay_factor=False,
        mask_out=False,
        use_stochastic_depth=False,
        drop_rate=0.0,
        schedule_mode=schedule_mode,
        lr_drop=None,
        lr_drop_rate=0.1,
        keep_first_quant=False,
        keep_last_quant=False,
        remove_residual_detach=False,
        use_out_phi=False,
        use_out_phi_res=False,
        use_lecam_reg=False,
        lecam_weight=0.05,
        perceptual_model='vgg16',
        base_ch_disc=64,
        random_flip=False,
        flip_prob=0.5,
        flip_mode='stochastic',
        max_flip_lvl=1,
        not_load_optimizer=False,
        use_lecam_reg_zero=False,
        freeze_encoder=False,
        rm_downsample=False,
        random_flip_1lvl=False,
        flip_lvl_idx=0,
        drop_when_test=False,
        drop_lvl_idx=0,
        drop_lvl_num=1,
        disc_version='v1',
        magvit_disc=False,
        sigmoid_in_disc=False,
        activation_in_disc='leaky_relu',
        apply_blur=False,
        apply_noise=False,
        dis_warmup_steps=0,
        dis_lr_multiplier=1.0,
        dis_minlr_multiplier=False,
        disc_channels=64,
        disc_layers=3,
        discriminator_iter_start=0,
        disc_pretrain_iter=0,
        disc_optim_steps=1,
        disc_warmup=0,
        disc_pool='no',
        disc_pool_size=1000,
        advanced_disc=False,
        recon_loss_type='l1',
        video_perceptual_weight=0.0,
        image_gan_weight=1.0,
        video_gan_weight=1.0,
        image_disc_weight=0.0,
        video_disc_weight=0.0,
        l1_weight=4.0,
        gan_feat_weight=0.0,
        perceptual_weight=0.0,
        kl_weight=0.0,
        lfq_weight=0.0,
        entropy_loss_weight=0.1,
        commitment_loss_weight=0.25,
        diversity_gamma=1,
        norm_type='group',
        disc_loss_type='hinge',
        use_checkpoint=False,
        precision='fp32',
        encoder_dtype='fp32',
        upcast_attention='',
        upcast_tf32=False,
        tokenizer='flux',
        pretrained=None,
        pretrained_mode='full',
        inflation_pe=False,
        init_vgen='no',
        no_init_idis=False,
        init_idis='keep',
        init_vdis='no',
        enable_nan_detector=False,
        turn_on_profiler=False,
        profiler_scheduler_wait_steps=10,
        debug=True,
        video_logger=False,
        bytenas='',
        username='',
        seed=1234,
        vq_to_vae=False,
        load_not_strict=False,
        zero=0,
        bucket_cap_mb=40,
        manual_gc_interval=1000,
        data_path=[''],
        data_type=[''],
        dataset_list=['imagenet'],
        fps=-1,
        dataaug='resizecrop',
        multi_resolution=False,
        random_bucket_ratio=0.0,
        sequence_length=16,
        resolution=[256, 256],
        batch_size=[1],
        num_workers=0,
        image_channels=3,
        codebook_size=codebook_size,
        codebook_l2_norm=True,
        codebook_show_usage=True,
        commit_loss_beta=0.25,
        entropy_loss_ratio=0.0,
        base_ch=128,
        num_res_blocks=2,
        encoder_ch_mult=encoder_ch_mult,
        decoder_ch_mult=decoder_ch_mult,
        dropout_p=0.0,
        cnn_type='2d',
        cnn_version='v1',
        conv_in_out_2d='no',
        conv_inner_2d='no',
        res_conv_2d='no',
        cnn_attention='no',
        cnn_norm_axis='spatial',
        flux_weight=0,
        cycle_weight=0,
        cycle_feat_weight=0,
        cycle_gan_weight=0,
        cycle_loop=0,
        z_drop=0.0)
    
    vae = AutoEncoder(args)

    if args.vqgan_ckpt:
        use_vae = vae.use_vae
        if not use_vae:
            num_codes = args.codebook_size
        if isinstance(vqgan_ckpt, str):
            state_dict = torch.load(args.vqgan_ckpt, map_location=torch.device("cpu"), weights_only=True)
        else:
            state_dict = args.vqgan_ckpt
        if state_dict:
            if args.ema == "yes":
                vae, new_state_dict, loaded_keys = load_cnn(vae, state_dict["ema"], prefix="", expand=False)
            else:
                vae, new_state_dict, loaded_keys = load_cnn(vae, state_dict["vae"], prefix="", expand=False)
    if test_mode:
        vae.eval()
        [p.requires_grad_(False) for p in vae.parameters()]
    return vae