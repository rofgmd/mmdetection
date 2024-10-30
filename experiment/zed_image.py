import pyzed.sl as sl
import numpy as np
from mmdet.apis import DetInferencer
import mmcv
import cv2
from pycocotools import mask as mask_utils
import random
import time

# 生成随机颜色
def random_color():
    return [random.randint(0, 255) for _ in range(3)]

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
    # 初始化MMDetection模型
    # inferencer = DetInferencer(model="/home/kevin/mmdetection/configs/cow/rtmdet-ins_s_1xb4-300e_cow_update_20241017.py",
    #                            weights="/home/kevin/mmdetection/work_dirs/rtmdet-ins_s_1xb4-300e_cow_update_20240603/epoch_300.pth", 
    #                            device='cuda:0',
    #                            show_progress=False)  
    inferencer = DetInferencer(model="/home/kevin/mmdetection/configs/cow/rtmdet-ins_tiny_1xb4-500e_cow_update_20241017.py",
                               weights="/home/kevin/mmdetection/work_dirs/rtmdet-ins_tiny_1xb4-500e_cow_update_20241017/epoch_300.pth", 
                               device='cuda:0',
                               show_progress=False)  
    # inferencer = DetInferencer(model="/home/kevin/mmdetection/configs/cow/mask-rcnn_r50_fpn_1x_cow_update_1005.py",
    #                            weights="/home/kevin/mmdetection/work_dirs/mask-rcnn_r50_fpn_1x_cow_update_0603/epoch_100.pth", 
    #                            device='cuda:0',
    #                            show_progress=False)  
    # inferencer = DetInferencer(model="/home/kevin/mmdetection/configs/cow/solov2_r50_fpn_ms-3x_cow_update_1005.py",
    #                            weights="/home/kevin/mmdetection/work_dirs/solov2_r50_fpn_ms-3x_cow_update_0602/epoch_150.pth", 
    #                            device='cuda:0',
    #                            show_progress=False)         
    # inferencer = DetInferencer(model="/home/kevin/mmdetection/configs/cow/solo_r50_fpn_1x_cow_update_1005.py",
    #                            weights="/home/kevin/mmdetection/work_dirs/solo_r50_fpn_1x_cow_update_0602/epoch_150.pth", 
    #                            device='cuda:0',
    #                            show_progress=False)         
    # inferencer = DetInferencer(model="/home/kevin/mmdetection/configs/cow/yolact_r50_1xb2-150e_cow_update_1005.py",
    #                            weights="/home/kevin/mmdetection/work_dirs/yolact_r50_1xb2-150e_cow_update_0602/epoch_150.pth", 
    #                            device='cuda:0',
    #                            show_progress=False)  

    # 读取并处理每一帧
    frame_idx = 0  # 用于命名掩码文件
    process_time = []
    while True:
        if cam.grab() == sl.ERROR_CODE.SUCCESS:
            # 获取左眼图像
            start_time = time.time()
            cam.retrieve_image(image, sl.VIEW.LEFT)

            # 将图像数据转换为NumPy格式
            image_np = image.get_data()

            # 检查图像格式是否为BGR
            if image_np.shape[2] == 4:  # ZED SDK可能返回BGRA图像，需要去掉Alpha通道
                image_np = image_np[:, :, :3]

            # 进行实例分割推理
            results = inferencer(image_np)

            # 获取预测结果
            predictions = results['predictions']
            end_time = time.time()
            process_time.append(end_time-start_time)
        else:
            break  # 到达SVO文件结尾

    print(f"average image process time is : {sum(process_time)/len(process_time)}")
    # 关闭相机
    cam.close()

if __name__ == "__main__":
    main()    