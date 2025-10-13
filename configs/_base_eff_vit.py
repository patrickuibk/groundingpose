_base_ = '_base.py'

model = dict(
    backbone=dict(
        type='EfficientViTBackbone',
        name='efficientvit_backbone_l0',
        out_indices=(2, 3, 4),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l0.pt'
        )
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[128, 256, 512],
    )   
)