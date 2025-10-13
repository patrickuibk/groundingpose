_base_ = '_base.py'


dataset_type = 'CocoStylePoseDataset'
data_root = './data/mp100'

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/mp100_split1_train_updated.json',
        data_root=data_root,
        consistent_keypoints_per_category=False,
        only_visible_keypoints=True,
        data_prefix=dict(img='images')))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='annotations/mp100_split1_test_updated.json',
        data_root=data_root,
        consistent_keypoints_per_category=False,
        only_visible_keypoints=True,
        data_prefix=dict(img='images')))

test_dataloader = val_dataloader


val_evaluator = dict(
    type='GroupedPCKAccuracy',
    thr = 0.2
)

test_evaluator = val_evaluator

max_epoch = 100

train_cfg = dict(
    max_epochs=max_epoch, 
    val_interval=5)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[90],
        gamma=0.1)
]