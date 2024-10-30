from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env
import cProfile
import pstats
import mmdet

def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMDetection'] = f'{mmdet.__version__}+{get_git_hash()[:7]}'
    return env_info

def get_depth_weight(depth_value):
    # 根据深度值的范围，设置权重
    return np.where(depth_value < 4, 1.0, np.where(depth_value < 6, 0.8, 0))

if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')

from mmdet.apis import DetInferencer
# inferencer = DetInferencer(model="/home/kevin/mmdetection/configs/cow/rtmdet-ins_s_1xb4-300e_cow_update_20240327.py",weights="/home/kevin/mmdetection/work_dirs/rtmdet-ins_s_1xb4-300e_cow_update_20240327/epoch_460.pth", device='cuda:0')# using rtm-ins-s
# inferencer = DetInferencer(model="/home/kevin/mmdetection/configs/cow/yolact_r50_1xb2-150e_cow_update_0602.py",weights="/home/kevin/mmdetection/work_dirs/yolact_r50_1xb2-150e_cow_update_0602/epoch_150.pth", device='cuda:0')# using yolact
# inferencer = DetInferencer(model="/home/kevin/mmdetection/configs/cow/solo_r50_fpn_1x_cow_update_0602.py",weights="/home/kevin/mmdetection/work_dirs/solo_r50_fpn_1x_cow_update_0602/epoch_150.pth", device='cuda:0')# using solo
# inferencer = DetInferencer(model="/home/kevin/mmdetection/configs/cow/mask-rcnn_r50_fpn_1x_cow_update_0603.py",weights="/home/kevin/mmdetection/work_dirs/mask-rcnn_r50_fpn_1x_cow_update_0603/epoch_100.pth", device='cuda:0')# using mask-rcnn
inferencer = DetInferencer(model="/home/kevin/mmdetection/configs/cow/rtmdet-ins_s_1xb4-300e_cow_update_20240603.py",weights="/home/kevin/mmdetection/work_dirs/rtmdet-ins_s_1xb4-300e_cow_update_20240603/epoch_300.pth", device='cuda:0')# using new-rtm-ins-s
# inferencer = DetInferencer(model="/home/kevin/mmdetection/configs/cow/decoupled-solo-light_r50_fpn_3x_cow_update_0602.py",weights="/home/kevin/mmdetection/work_dirs/decoupled-solo-light_r50_fpn_3x_cow_update_0602/best_coco_segm_mAP_50_epoch_101.pth", device='cuda:0')
# inferencer = DetInferencer(model="/home/kevin/mmdetection/configs/cow/rtmdet-ins_m_1xb4-300e_cow_update_20240604.py",weights="/home/kevin/mmdetection/work_dirs/rtmdet-ins_m_1xb4-300e_cow_update_20240604/best_coco_segm_mAP_50_epoch_291.pth", device='cuda:0')
# inferencer = DetInferencer(model="/home/kevin/mmdetection/configs/cow/solov2_r50_fpn_ms-3x_cow_update_0602.py",weights="/home/kevin/mmdetection/work_dirs/solov2_r50_fpn_ms-3x_cow_update_0602/best_coco_segm_mAP_epoch_141.pth", device='cuda:0')

from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import cv2  # For image loading
import os
import time
import re
from pycocotools.mask import decode
from PIL import Image
import json

def main():
    coco = COCO('/home/kevin/mmdetection/data/cow_update_20240325/annotations/val.json')
    # coco = COCO('/home/kevin/mmdetection/data/cow_update_20240325/annotations/train.json')
    # RGB和深度图像的目录
    rgb_dir = '/home/kevin/mmdetection/data/cow_update_20240325/val2017'
    # rgb_dir = '/home/kevin/mmdetection/data/cow_update_20240325/train2017'
    # depth_dir = '/mnt/h/labelme/ZED标记图片/output_folder'
    depth_dir = '/home/kevin/mmdetection/data/cow_update_20240325/depth_image'

    # Initialize counters for area_ratio < 1 and > 1
    count_less_than_one = 0
    count_greater_than_one = 0

    ratio=[]
    full_list=[]
    process_time = []
    # 遍历RGB目录下的所有文件
    for rgb_filename in os.listdir(rgb_dir):
        # 从RGB文件名中提取数字
        match = re.search(r'(\d+)', rgb_filename)
        if match:
            num = match.group(1)
            # 构造深度图文件名
            depth_filename = f'depth{int(num):06}.png'
            depth_filepath = os.path.join(depth_dir, depth_filename)
            if os.path.exists(depth_filepath):
                # 执行你的处理流程
                rgb_filepath = os.path.join(rgb_dir, rgb_filename)
                start_pred_time = time.time()
                predictions = inferencer(rgb_filepath, out_dir='/home/kevin/mmdetection/output/cow_fodder_bunk_output')
                end_pred_time = time.time()
                process_time.append(end_pred_time-start_pred_time)
                # 接下来，根据你的代码使用predictions和depth_filepath进行处理

                # 注意: 以下部分需要根据你的具体需要进行调整，这里仅提供一个示范性的流程
                # 加载深度图像
                depth_image = cv2.imread(depth_filepath, cv2.IMREAD_UNCHANGED)
                # 你的处理代码...
                predictions=predictions['predictions'][0]
                labels = predictions['labels']
                scores = predictions['scores']
                masks = predictions['masks']
                bboxes = predictions['bboxes']

                # 初始化存储label=1对应的masks的列表
                mask_bunk = []
                # 遍历labels列表
                for index, label in enumerate(labels):
                    # 检查当前label是否为1
                    if label == 1:
                        # 如果是，将对应的mask添加到mask_bunk列表中
                        mask_bunk.append(masks[index])

                # 初始化存储label=1对应的masks的列表
                mask_fodder = []
                # 遍历labels列表
                for index, label in enumerate(labels):
                    # 检查当前label是否为1
                    if label == 3:
                        # 如果是，将对应的mask添加到mask_bunk列表中
                        mask_fodder.append(masks[index])

                # 假设mask_fodder包含了我们想要解码和显示的6个masks的信息

                # 初始化一个空的数组用于叠加所有掩码，确保数组的大小与掩码尺寸匹配
                # 假设所有掩码尺寸相同，使用mask_fodder中第一个掩码的尺寸初始化
                if len(mask_bunk) == 0:
                    continue
                mask_size = mask_bunk[0]['size']
                bunk_mask = np.zeros((mask_size[0], mask_size[1]), dtype=np.uint8)

                # 遍历所有masks
                for mask_info in mask_bunk:
                    # 从mask_info中获取RLE编码的mask和尺寸
                    rle_mask = mask_info['counts']
                    # RLE编码的mask需要是一个字典，其中'counts'是编码后的字符串，'size'是mask的尺寸
                    rle = {
                        'counts': rle_mask,
                        'size': mask_size
                    }
                    
                    # 使用pycocotools的decode函数解码RLE mask
                    decoded_mask = decode(rle)
                    
                    # 将解码后的掩码叠加到combined_mask上
                    # 注意: 这里简单地将它们相加。如果掩码重叠，可能需要根据具体情况调整合并方式
                    bunk_mask |= decoded_mask

                fodder_area = np.sum(bunk_mask)
                print(fodder_area)
                bunk_depth = np.where(bunk_mask, depth_image, 0)

                # 假设mask_fodder包含了我们想要解码和显示的6个masks的信息

                # 初始化一个空的数组用于叠加所有掩码，确保数组的大小与掩码尺寸匹配
                # 假设所有掩码尺寸相同，使用mask_fodder中第一个掩码的尺寸初始化
                if len(mask_fodder) == 0:
                    continue
                mask_size = mask_fodder[0]['size']
                fodder_mask = np.zeros((mask_size[0], mask_size[1]), dtype=np.uint8)

                # 遍历所有masks
                for mask_info in mask_fodder:
                    # 从mask_info中获取RLE编码的mask和尺寸
                    rle_mask = mask_info['counts']
                    # RLE编码的mask需要是一个字典，其中'counts'是编码后的字符串，'size'是mask的尺寸
                    rle = {
                        'counts': rle_mask,
                        'size': mask_size
                    }
                    
                    # 使用pycocotools的decode函数解码RLE mask
                    decoded_mask = decode(rle)
                    
                    # 将解码后的掩码叠加到combined_mask上
                    # 注意: 这里简单地将它们相加。如果掩码重叠，可能需要根据具体情况调整合并方式
                    # fodder_mask |= decoded_mask
                    fodder_mask = np.bitwise_or(fodder_mask, decoded_mask)

                fodder_area = np.sum(fodder_mask)
                print(fodder_area)
                fodder_depth = np.where(fodder_mask, depth_image, 0)

                # 累加深度值
                fodder_depth_sum = np.sum(fodder_depth*fodder_depth* get_depth_weight(fodder_depth))
                bunk_depth_sum = np.sum(bunk_depth*bunk_depth* get_depth_weight(bunk_depth))
                # 计算深度值之和的比例
                area_ratio = fodder_depth_sum / bunk_depth_sum if bunk_depth_sum > 0 else 0
                ratio.append(area_ratio)
                if area_ratio > 1:
                    full_list.append(rgb_filename)
                    # 打印 fodder_depth 的最小值、最大值和平均值
                    print("Fodder Depth Min:", np.min(fodder_depth))
                    print("Fodder Depth Max:", np.max(fodder_depth))
                    print("Fodder Depth Mean:", np.mean(fodder_depth))

                    # 打印 bunk_depth 的最小值、最大值和平均值
                    print("Bunk Depth Min:", np.min(bunk_depth))
                    print("Bunk Depth Max:", np.max(bunk_depth))
                    print("Bunk Depth Mean:", np.mean(bunk_depth))

    for area_ratio in ratio:
        if area_ratio < 1:
            count_less_than_one += 1
        elif area_ratio > 1:
            count_greater_than_one += 1
    # Print the counts
    print(f"Number of area_ratio < 1: {count_less_than_one}")
    print(f"Number of area_ratio > 1: {count_greater_than_one}")
    print(full_list)
    print(f"average image process time is : {sum(process_time)/len(process_time)}")

if __name__ == '__main__':
    main()