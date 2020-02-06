import time
import tensorflow as tf
import numpy as np
import cv2
import sys
import SDConfig

# 获取参数文件
CFG = SDConfig.cfg

sys.path.append(CFG.SDCSDK.SDKPATH)

from SDClassV3 import LaneNet, LaneContours, BirdsEyeView, YOLOV3TF


# rawMedia='test.jpg'
# rawMedia = 'test_videos/output.mp4'
rawMedia = 'test_videos/myraw4.mp4'

width, height = (CFG.FRAME.WIDTH, CFG.FRAME.HEIGHT)

# instance
ln = LaneNet(CFG.LANENET.MODELPATH)

laneContours = LaneContours(width, height)

birdView = BirdsEyeView()

yoloTF=YOLOV3TF()

# 加载yolo gpu模型
yoloTF.yoloLoadModule(CFG.YOLOV3.className, CFG.YOLOV3.isTiny, CFG.YOLOV3.gpu_memory_fraction, CFG.YOLOV3.ckpt_file, CFG.YOLOV3.conf_threshold, CFG.YOLOV3.iou_threshold, CFG.YOLOV3.data_format)


# 鸟瞰图功能初始化
birdView.init(width, height, CFG.BIRDVIEW.upInsideOffset,
              CFG.BIRDVIEW.downOutsideOffset, CFG.BIRDVIEW.horizonUnderHalfHeight)


def nothing(emp):
    pass

def processingAll(img):

    img = cv2.resize(img, (width, height))

    cv2.imshow('img', img)

    image_binarized = ln.binarize(img)

    rawWarped=birdView.getWarp(img, False)
    # cv2.imshow('rawWarped', rawWarped)

    binaryWarped=birdView.getWarp(image_binarized*255,False)

    morphologicalBirdView = laneContours.morphological_process(binaryWarped, kernel_size=15)
    # cv2.imshow('morphologicalBirdView', morphologicalBirdView)
    
    # 获得轮廓处理后的车道线图层
    contoursLayerBirdView = birdView.getBirdViewLaneContoursLayer(morphologicalBirdView, 70, True)[0]
    # cv2.imshow('contoursLayerBirdView', contoursLayerBirdView)

    filtered_boxes=yoloTF.yoloGetBoxs(img)
    finalBoxs=yoloTF.getFinalObjBoxs(filtered_boxes, width, height, yoloTF.classes, (CFG.YOLOV3.size, CFG.YOLOV3.size), True)
    birdObjsLayer=birdView.getBirdViewObjDistLayer(finalBoxs, width, height, isDistance=False)

    infoLayer = cv2.addWeighted(contoursLayerBirdView, 1.0, birdObjsLayer, 1.0, 0)
    combin = cv2.addWeighted(rawWarped, 0.3, infoLayer, 1.0, 0)

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
