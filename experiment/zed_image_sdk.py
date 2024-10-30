import pyzed.sl as sl
import numpy as np
from pycocotools import mask as mask_utils
import random
import time
import torch
import os
import cv2
from mmdeploy_runtime import Detector

def main():
    # 创建ZED相机对象
    cam = sl.Camera()

    # 初始化
    init_params = sl.InitParameters()
    init_params.set_from_svo_file("/home/kevin/mmdetection/demo/demo_svo/1.svo")  # 设置SVO文件路径
    # init_params.svo_real_time_mode = False  # 非实时模式，SVO文件按自己的速度处理
    init_params.svo_real_time_mode = True  # 实时模式
    # 打开相机
    err = cam.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error: {err}")
        exit(1)

    # 准备图像容器
    image = sl.Mat()   
    model_path = '/home/kevin/mmdeploy/mmdeploy_models/cow/rtmdet-ins_tiny'     
    detector = Detector(model_path, device_name = 'cuda', device_id=0)

    # 读取并处理每一帧
    process_time = []
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
            start_time = time.time()
            bboxes, labels, masks = detector(image_np)
            end_time = time.time()
            predict_time = end_time-start_time
            # 计算分割效率
            process_time.append(predict_time)
        else:
            break  # 到达SVO文件结尾

    print(f"average image process time is : {sum(process_time)/len(process_time)}")
    # 关闭相机
    cam.close()

if __name__ == "__main__":
    main()    