from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch

deploy_cfg = '/home/kevin/mmdeploy/configs/mmdet/instance-seg/instance-seg_rtmdet-ins_onnxruntime_static-640x640.py'
# model_cfg = 'configs/cow/rtmdet-ins_tiny_1xb4-500e_cow_update_20240416.py' # 实验室分割权重
model_cfg = '/home/kevin/mmdetection/configs/cow/rtmdet-ins_s_1xb4-300e_cow_update_20241017.py' # 奶牛厂环境
device = 'cuda'
backend_model = ['/home/kevin/mmdetection/work_dirs/rtmdet-ins_s_1xb4-300e_cow_update_20240603/end2end.onnx']
image = '/home/kevin/mmdetection/demo/demo_image/963.jpg'

# read deploy_cfg and model_cfg
deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

# build task and backend model
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_model)

# process input image
input_shape = get_input_shape(deploy_cfg)
model_inputs, _ = task_processor.create_input(image, input_shape)

# do model inference
with torch.no_grad():
    result = model.test_step(model_inputs)

# visualize results
task_processor.visualize(
    image=image,
    model=model,
    result=result[0],
    window_name='visualize',
    output_file='output/test_onnx/outdoor_experiment.png')