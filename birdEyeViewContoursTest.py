import time
import tensorflow as tf
import numpy as np
import cv2
import sys
import SDConfig

# 获取参数文件
CFG = SDConfig.cfg
sys.path.append(CFG.SDCSDK.SDKPATH)
# 引入车道线,鸟瞰图,轮廓包
from SDClassV3 import LaneNet, LaneContours, BirdsEyeView


# rawMedia='test.jpg'
rawMedia = 'test_videos/myraw4.mp4'

width, height = (CFG.FRAME.WIDTH, CFG.FRAME.HEIGHT)

# 定义instance对象
ln = LaneNet(CFG.LANENET.MODELPATH)
# 定义轮廓对象
laneContours = LaneContours(width, height)
# 定义鸟瞰图对象
birdView = BirdsEyeView()
# 鸟瞰图功能初始化
birdView.init(width, height, CFG.BIRDVIEW.upInsideOffset,
              CFG.BIRDVIEW.downOutsideOffset, CFG.BIRDVIEW.horizonUnderHalfHeight)


def nothing(emp):
    pass

def processingAll(img):
    # 调整图片大小
    img = cv2.resize(img, (width, height))
    # 显示原始图象 
    cv2.imshow('img', img)
    # 车道线二值化处理
    image_binarized = ln.binarize(img)
    # 生成原图的鸟瞰图
    rawWarped=birdView.getWarp(img, False)
    # cv2.imshow('rawWarped', rawWarped)
    # 生成车道线二值图的鸟瞰图层
    binaryWarped=birdView.getWarp(image_binarized*255,False)
    # 图像形态学处理，生成车道线图层
    morphologicalBirdView = laneContours.morphological_process(binaryWarped, kernel_size=15)
    # cv2.imshow('morphologicalBirdView', morphologicalBirdView)
    # 获得轮廓处理后的车道线图层
    contoursLayerBirdView = birdView.getBirdViewLaneContoursLayer(morphologicalBirdView, 70, True)[0]
    # cv2.imshow('contoursLayerBirdView', contoursLayerBirdView)
    # 合成两个处理后的图层
    combin = cv2.addWeighted(rawWarped, 0.3, contoursLayerBirdView, 1.0, 0)
    # 显示最终结果
    cv2.imshow('combin', combin)

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
