_base_ = '_base.py'


dataset_type = 'CocoStylePoseDataset'
data_root = 'data/crowd_pose'

relation_name = 'belongs to'


train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        relation_name=relation_name,
        only_visible_keypoints=True,
        sigmas='0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.079, 0.079',
        ann_file='annotations/crowdpose_trainval.json',
        data_root=data_root,
        data_prefix=dict(img='images')))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        relation_name=relation_name,
        ann_file='annotations/crowdpose_test.json',
        sigmas='0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.079, 0.079',
        data_root=data_root,
        data_prefix=dict(img='images')))

test_dataloader = val_dataloader


val_evaluator = dict(
    type='PoseCocoMetric',
    ann_file=data_root + '/annotations/crowdpose_test.json',
    use_area=False,
    iou_type='keypoints_crowd',
    nms_mode='none',
    outfile_prefix='data/crowd_pose/outputs/preds',
    node_score_thresh=0.3,
    edge_score_thresh=0.3
)

test_evaluator = val_evaluator

max_epoch = 200

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
        milestones=[180],
        gamma=0.1)
]