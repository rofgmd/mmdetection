import pyzed.sl as sl
import numpy as np
from mmdet.apis import DetInferencer
import cv2
import threading
import time
import queue

def init():
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
    inferencer = DetInferencer(model="/home/kevin/mmdetection/configs/cow/rtmdet-ins_tiny_1xb4-500e_cow_update_20241017.py",
                               weights="/home/kevin/mmdetection/work_dirs/rtmdet-ins_tiny_1xb4-500e_cow_update_20241017/epoch_300.pth", 
                               device='cuda:0',
                               show_progress=False)  

    return cam, image, inferencer 

# 图像读取线程函数
def image_reader(cam, frame_queue, batch_size, stop_event):
    image = sl.Mat()
    while not stop_event.is_set():
        if cam.grab() == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(image, sl.VIEW.LEFT)
            image_np = image.get_data()

            if image_np.shape[2] == 4:  # 去掉Alpha通道
                image_np = image_np[:, :, :3]

            # 将图像添加到队列中
            frame_queue.put(image_np)

        else:
            # 到达视频文件结尾，停止读取
            stop_event.set()
            break


# 推理处理线程函数
def inference_worker(inferencer, frame_queue, result_queue, batch_size, stop_event):
    while not stop_event.is_set() or not frame_queue.empty():
        batch_frames = []
        
        # 从队列中读取帧图像，直到收集到一个批次
        while len(batch_frames) < batch_size and not frame_queue.empty():
            try:
                frame = frame_queue.get(timeout=1)  # 设置超时避免线程死锁
                batch_frames.append(frame)
            except queue.Empty:
                continue  # 如果队列为空则继续等待

        if batch_frames:
            # 执行推理
            start_time = time.time()
            results = inferencer(batch_frames)
            predictions_batch = results['predictions']
            end_time = time.time()

            # 计算每张图片的处理时间
            process_time_per_image = (end_time - start_time) / batch_size

            # 将结果存储到结果队列中
            result_queue.put((predictions_batch, process_time_per_image))


def main():
    cam, image, inferencer = init()
    # 设置队列和批处理大小
    frame_queue = queue.Queue(maxsize=10)  # 存储帧的队列
    result_queue = queue.Queue()  # 存储推理结果的队列
    batch_size = 5  # 一次处理的帧数
    process_time = []    
    
    # 事件标志，用于停止线程
    stop_event = threading.Event()

    # 启动图像读取线程
    reader_thread = threading.Thread(target=image_reader, args=(cam, frame_queue, batch_size, stop_event))
    reader_thread.start()

    # 启动推理处理线程
    inference_thread = threading.Thread(target=inference_worker, args=(inferencer, frame_queue, result_queue, batch_size, stop_event))
    inference_thread.start()

    # 等待所有线程完成
    reader_thread.join()
    inference_thread.join()

    # 处理推理结果
    process_time_per_image_list = []
    while not result_queue.empty():
        predictions_batch, infer_time_per_image = result_queue.get()
        process_time_per_image_list.append(infer_time_per_image)
        print(f"Processed a batch, each image took {infer_time_per_image:.4f} seconds")

    if process_time_per_image_list:  # 避免空列表
        average_time_per_image = sum(process_time_per_image_list) / len(process_time_per_image_list)
        print(f"Average image process time is: {average_time_per_image:.4f} seconds")

    # 关闭相机
    cam.close()

if __name__ == "__main__":
    main()
    # md，多线程优化还不如顺序处理呢