_base_ = '_base.py'

dataset_type = 'CocoStyleTopDownPoseDataset'
data_root = './data/mp100'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadKeypointGraphAnnotationsAsBbox'),
    dict(type='TopDownBBoxCrop'),
    dict(type='Rotate', prob=0.5, level=1, max_mag=30.0),
    dict(type='Brightness', prob=0.5, level=1),
    dict(type='Contrast', prob=0.5, level=1),
    dict(type='RandomChoiceResize',
         scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                 (736, 1333), (768, 1333), (800, 1333)],
         keep_ratio=True),
    dict(type='ConvertRelationsToMatrix'),
    dict(type='TransformKeypoints'),
    dict(type='PackKeypointGraphInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction', 'text',
                    'relation_text', 'custom_entities'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='TopDownBBoxCrop'),
    dict(type='FixScaleResize', scale=(800, 1333), keep_ratio=True),
    dict(type='LoadKeypointGraphAnnotationsAsBbox'),
    dict(type='ConvertRelationsToMatrix'),
    dict(type='PackKeypointGraphInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'text', 'relation_text', 'custom_entities',
                    'crowd_index', 'raw_ann_info'))
]

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/mp100_split1_train_updated.json',
        data_root=data_root,
        consistent_keypoints_per_category=False,
        only_visible_keypoints=True,
        data_prefix=dict(img='images'),
        pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/mp100_split1_test_updated.json',
        data_root=data_root,
        consistent_keypoints_per_category=False,
        only_visible_keypoints=True,
        data_prefix=dict(img='images'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='TopDownPCKAccuracy', thr=0.2)
test_evaluator = val_evaluator

max_epoch = 50
train_cfg = dict(max_epochs=max_epoch, val_interval=5)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(type='MultiStepLR', begin=0, end=max_epoch, by_epoch=True, milestones=[40], gamma=0.1)
]