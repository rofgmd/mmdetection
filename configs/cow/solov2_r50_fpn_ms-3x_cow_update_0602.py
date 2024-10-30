# 新配置继承了基本配置，并做了必要的修改
_base_ = '../solov2/solov2_r50_fpn_ms-3x_coco.py'

# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
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

max_epochs = 150
interval = 10
# max_epochs
train_cfg = dict(max_epochs=max_epochs)

# hooks
default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=1,  # only keep latest 1 checkpoints
        save_best='coco/segm_mAP'
    ))

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
test_evaluator = val_evaluator

# 使用预训练的 Yolact 模型权重来做初始化，可以提高模型性能
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/solov2/solov2_r50_fpn_3x_coco/solov2_r50_fpn_3x_coco_20220512_125856-fed092d4.pth'