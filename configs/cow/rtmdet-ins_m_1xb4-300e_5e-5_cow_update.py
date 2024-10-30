# 新配置继承了基本配置，并做了必要的修改
_base_ = './rtmdet-ins_m_1xb4-300e_cow_update.py'

max_epochs=300
base_lr=0.00005
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05))

# max_epochs
train_cfg = dict(max_epochs=max_epochs)

load_from = '/home/kevin/mmdetection/work_dirs/rtmdet-ins_m_1xb4-500e_cow_update/epoch_330.pth'