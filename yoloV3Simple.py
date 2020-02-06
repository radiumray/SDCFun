import time
import tensorflow as tf
import numpy as np
import cv2
import sys
import SDConfig

# 获取参数文件
CFG = SDConfig.cfg
sys.path.append(CFG.SDCSDK.SDKPATH)
# 引入yoloV3包
from SDClassV3 import YOLOV3TF

# rawMedia='test.jpg'
# rawMedia = 'test_videos/output.mp4'
rawMedia = 'test_videos/myraw4.mp4'

width, height = (CFG.FRAME.WIDTH, CFG.FRAME.HEIGHT)
# 定义yoloV3对象
yoloTF=YOLOV3TF()
# 加载yolo gpu模型
yoloTF.yoloLoadModule(CFG.YOLOV3.className, CFG.YOLOV3.isTiny, CFG.YOLOV3.gpu_memory_fraction, CFG.YOLOV3.ckpt_file, CFG.YOLOV3.conf_threshold, CFG.YOLOV3.iou_threshold, CFG.YOLOV3.data_format)

def nothing(emp):
    pass

def processingAll(img):
    # 获取原始图象
    img = cv2.resize(img, (width, height))
    # 显示原始图象
    cv2.imshow('img', img)
    # 推理yoloV3模型,输入原始图象，输出bboxs的坐标(针对yolo模型的输入大小416，416的坐标)
    filtered_boxes=yoloTF.yoloGetBoxs(img)
    # 对bboxs坐标进行转换,iou,mns后续处理，并在图像中画出
    finalImg=yoloTF.draw_boxes(filtered_boxes, img, yoloTF.classes, (CFG.YOLOV3.size, CFG.YOLOV3.size), True)
    # 显示处理后的图像
    cv2.imshow('finalImg', finalImg)

if(rawMedia.endswith('.mp4')):

    cv2.namedWindow('processBar')
    cv2.resizeWindow('processBar', 400,50)

    loop_flag = 0
    pos = 0
    cap = cv2.VideoCapture(rawMedia)
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.createTrackbar('time', 'processBar', 0, totalFrames, nothing)

    while(cap.isOpened()):

        if loop_flag == pos:
            loop_flag = loop_flag + 1
            cv2.setTrackbarPos('time', 'processBar', loop_flag)
        else:
            pos = cv2.getTrackbarPos('time', 'processBar')
            loop_flag = pos
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        _, img = cap.read()
        if img is None:
            break

        processingAll(img)

        # time.sleep(0.001)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.waitKey(0)
        # if cv2.waitKey(1) & 0xFF == ord('r'):
        #    cv2.imwrite('check1.jpg', undist_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

else:
    img = cv2.imread(rawMedia, cv2.IMREAD_COLOR)
    processingAll(img)
    cv2.waitKey(0)
