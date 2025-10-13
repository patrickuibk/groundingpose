_base_ = 'mp100split1.py'


train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/mp100_split5_train_updated.json'))

val_dataloader = dict(
    dataset=dict(
        ann_file='annotations/mp100_split5_test_updated.json'))

test_dataloader = val_dataloader
