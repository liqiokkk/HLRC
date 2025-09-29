_base_ = [
    '../_base_/models/upernet_swin_refine.py', '../_base_/datasets/aeril_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
checkpoint_file = 'pretrain/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    down_ratio=0.25,
    refine_input_ratio=1.0,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
        #with_cp=True),
    decode_head=[
        dict(
            type='UPerHeadRefine',
            in_channels=[96, 192, 384, 768],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='RegionHead',
            in_channels=192, # effvit
            channels=512,  # deep
            region_channels=192, #
            effvit='pretrain/b2-r224.pt',
            num_classes=2,
            sr_channels=None,
            expert_number=8,
            tok_k=2,
            norm_cfg=norm_cfg,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    auxiliary_head=dict(in_channels=384, num_classes=2),
    test_cfg=dict(mode='slide', crop_size=(1280, 1280), stride=(960, 960)))
    # test_cfg=dict(mode='whole'))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        # custom_keys={
            # 'absolute_pos_embed': dict(decay_mult=0.),
            # 'relative_position_bias_table': dict(decay_mult=0.),
            # 'norm': dict(decay_mult=0.)
        # }
        ))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(
    samples_per_gpu=6, # batch_size
    workers_per_gpu=8,)
