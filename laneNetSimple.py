import time
import tensorflow as tf
import numpy as np
import cv2
import sys
import SDConfig

# 获取参数文件
CFG = SDConfig.cfg
sys.path.append(CFG.SDCSDK.SDKPATH)

# 加载车道线类
from SDClassV3 import LaneNet

# 定义测试视频地址
# rawMedia='test.jpg'
rawMedia = 'test_videos/myraw4.mp4'

width, height = (CFG.FRAME.WIDTH, CFG.FRAME.HEIGHT)

# 定义instance对象
ln = LaneNet(CFG.LANENET.MODELPATH)

def nothing(emp):
    pass

def processingAll(img):
    # 调整图片大小
    img = cv2.resize(img, (width, height))
    # 显示原始图象 
    cv2.imshow('img', img)
    # 车道线二值化处理
    image_binarized = ln.binarize(img)
    # 显示车道线二值化处理图层
    cv2.imshow('image_binarized', image_binarized*255)

# 如果是视频
if(rawMedia.endswith('.mp4')):
    # 加载进度条
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
        # 模型推理处理函数
        processingAll(img)

        # time.sleep(0.001)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.waitKey(0)
        # if cv2.waitKey(1) & 0xFF == ord('r'):
        #    cv2.imwrite('check1.jpg', undist_img)
        # 输入q推出
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    cap.release()
    cv2.destroyAllWindows()

# 否则测试图片
else:
    # 读取图片
    img = cv2.imread(rawMedia, cv2.IMREAD_COLOR)
    # 模型推理处理函数
    processingAll(img)
    cv2.waitKey(0)
