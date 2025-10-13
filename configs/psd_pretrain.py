_base_ = '_base.py'


data_root = 'data/psd_pretrain'


model = dict(
    decoder=dict(
        num_layers=4
    ),
    test_cfg=dict(
        max_per_img=900,
    ))

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        data_root=data_root))

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/test')))

test_dataloader = val_dataloader


max_epoch = 500

train_cfg = dict(
    max_epochs=max_epoch, 
    val_interval=5)


default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto', rule='greater'),
    logger=dict(type='LoggerHook', interval=50))

optim_wrapper = dict(
    optimizer=dict(lr=0.0001),
    accumulative_counts=1,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=1.0),
            'language_model': dict(lr_mult=0),
        }))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[450],
        gamma=0.1)
]