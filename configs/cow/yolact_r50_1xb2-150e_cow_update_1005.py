# 新配置继承了基本配置，并做了必要的修改
_base_ = '../yolact/yolact_r50_1xb8-55e_coco.py'

# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    bbox_head=dict(num_classes=4),
    mask_head=dict(num_classes=4))

# 修改数据集相关配置
dataset_type = 'CocoDataset'
classes = ('cow.-buck.-fodder','bunk','cow','fodder',)
data_root = 'data/cow_update_20240325/'
image_path = data_root + 'images/'
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/')))
val_dataloader = dict( 
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=dict(classes=classes),
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/')))
test_dataloader = val_dataloader

interval = 10
# max_epochs
train_cfg = dict(max_epochs=150)

# hooks
default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'annotations/val.json', classwise = True)
test_evaluator = val_evaluator

# 使用预训练的 Yolact 模型权重来做初始化，可以提高模型性能
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/yolact/yolact_r50_1x8_coco/yolact_r50_1x8_coco_20200908-f38d58df.pth'