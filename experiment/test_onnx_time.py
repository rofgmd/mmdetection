from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch
import os
import time

def init(deploy_cfg, model_cfg, device, backend_model, rgb_dir):
    # read deploy_cfg and model_cfg
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
    # build task and backend model
    task_processor = build_task_processor(model_cfg, deploy_cfg, device)
    model = task_processor.build_backend_model(backend_model)
    # process input image
    input_shape = get_input_shape(deploy_cfg)
    return task_processor, model, input_shape

def inference_val_dataset(task_processor, model, input_shape, rgb_dir):
    process_time = []
    read_image_time = []
    pred_time = []
    for rgb_filename in os.listdir(rgb_dir):
        start_time = time.time()

        rgb_filepath = os.path.join(rgb_dir, rgb_filename)       
        model_inputs, _ = task_processor.create_input(rgb_filepath, input_shape)
        
        end_read_time = time.time()
        read_image_time.append(end_read_time-start_time)
        # do model inference
        with torch.no_grad():
            result = model.test_step(model_inputs)

        end_pred_time = time.time()
        pred_time.append(end_pred_time-end_read_time)
        process_time.append(end_pred_time-start_time)
    return process_time, read_image_time, pred_time

def main():
    # 可以用参数优化，但是没时间弄了
    deploy_cfg = '/home/kevin/mmdeploy/configs/mmdet/instance-seg/instance-seg_rtmdet-ins_onnxruntime_static-640x640.py'
    model_cfg = '/home/kevin/mmdetection/configs/cow/rtmdet-ins_s_1xb4-300e_cow_update_20241017.py' # 奶牛厂环境
    device = 'cuda'
    backend_model = ['/home/kevin/mmdetection/work_dirs/rtmdet-ins_s_1xb4-300e_cow_update_20240603/end2end.onnx']
    rgb_dir = '/home/kevin/mmdetection/data/cow_update_20240325/val2017'

    task_processor, model, input_shape = init(deploy_cfg, model_cfg, device, backend_model, rgb_dir)
    
    process_time, read_image_time, pred_time = inference_val_dataset(task_processor, model, input_shape, rgb_dir)
    print(f"average image read time is : {sum(read_image_time)/len(read_image_time)}")
    print(f"average image predict time is : {sum(pred_time)/len(pred_time)}")
    print(f"average image process time is : {sum(process_time)/len(process_time)}")

if __name__ == "__main__":
    main()