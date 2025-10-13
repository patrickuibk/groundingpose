_base_ = 'mp100split1_topdown.py'


train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/mp100_split2_train_updated.json'))

val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/mp100_split2_test_updated.json'))

test_dataloader = val_dataloader
