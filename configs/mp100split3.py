_base_ = 'mp100split1.py'


train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/mp100_split3_train_updated.json'))

val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/mp100_split3_test_updated.json'))

test_dataloader = val_dataloader
