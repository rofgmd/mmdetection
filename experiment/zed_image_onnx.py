import pyzed.sl as sl
import numpy as np
from pycocotools import mask as mask_utils
import random
import time
from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config
import torch
import os

# 生成随机颜色
def random_color():
    return [random.randint(0, 255) for _ in range(3)]

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

def inference_image(task_processor, model, input_shape, image_np):
    start_time = time.time()

    model_inputs, _ = task_processor.create_input(image_np, input_shape)
    
    end_read_time = time.time()
    read_image_time = end_read_time-start_time
    # do model inference
    with torch.no_grad():
        result = model.test_step(model_inputs)

    end_pred_time = time.time()
    pred_image_time = end_pred_time-end_read_time
    process_image_time = end_pred_time-start_time
    
    return read_image_time, pred_image_time, process_image_time, result[0]

def main():
    # 创建ZED相机对象
    cam = sl.Camera()

    # 初始化
    init_params = sl.InitParameters()
    init_params.set_from_svo_file("demo/demo_svo/1.svo")  # 设置SVO文件路径
    # init_params.svo_real_time_mode = False  # 非实时模式，SVO文件按自己的速度处理
    init_params.svo_real_time_mode = True  # 实时模式
    # 打开相机
    err = cam.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error: {err}")
        exit(1)

    # 准备图像容器
    image = sl.Mat()        

    # 可以用参数优化，但是没时间弄了
    deploy_cfg = '/home/kevin/mmdeploy/configs/mmdet/instance-seg/instance-seg_rtmdet-ins_tensorrt_static-640x640.py'
    # deploy_cfg = '/home/kevin/mmdeploy/configs/mmdet/instance-seg/instance-seg_onnxruntime_static.py'
    
    # model_cfg = '/home/kevin/mmdetection/configs/cow/rtmdet-ins_s_1xb4-300e_cow_update_20241017.py' # 奶牛厂环境
    model_cfg = '/home/kevin/mmdetection/configs/cow/rtmdet-ins_tiny_1xb4-500e_cow_update_20241017.py' # 奶牛厂环境
    # model_cfg = '/home/kevin/mmdetection/configs/cow/mask-rcnn_r50_fpn_1x_cow_update_1005.py' # 奶牛厂环境

    device = 'cuda'

    # backend_model = ['/home/kevin/mmdetection/work_dirs/rtmdet-ins_s_1xb4-300e_cow_update_20240603/end2end.onnx']
    backend_model = ['/home/kevin/mmdeploy/mmdeploy_models/cow/rtmdet-ins_tiny/end2end.engine']
    # backend_model = ['/home/kevin/mmdeploy/mmdeploy_models/cow/mask-rcnn-onnx-static/end2end.onnx']

    rgb_dir = '/home/kevin/mmdetection/data/cow_update_20240325/val2017'

    task_processor, model, input_shape = init(deploy_cfg, model_cfg, device, backend_model, rgb_dir)

    # 读取并处理每一帧
    frame_idx = 0  # 用于命名掩码文件
    process_time = []
    read_time = []
    pred_time = []
    while True:
        if cam.grab() == sl.ERROR_CODE.SUCCESS:
            # 获取左眼图像
            cam.retrieve_image(image, sl.VIEW.LEFT)

            # 将图像数据转换为NumPy格式
            image_np = image.get_data()

            # 检查图像格式是否为BGR
            if image_np.shape[2] == 4:  # ZED SDK可能返回BGRA图像，需要去掉Alpha通道
                image_np = image_np[:, :, :3]

            # 进行实例分割推理
            read_image_time, pred_image_time, process_image_time, result = inference_image(task_processor, model, input_shape, image_np)
            read_time.append(read_image_time)
            pred_time.append(pred_image_time)
            process_time.append(process_image_time)
        else:
            break  # 到达SVO文件结尾

    print(f"average image read time is : {sum(read_time)/len(read_time)}")
    print(f"average image predict time is : {sum(pred_time)/len(pred_time)}")
    print(f"average image process time is : {sum(process_time)/len(process_time)}")
    # 关闭相机
    cam.close()

if __name__ == "__main__":
    main()    