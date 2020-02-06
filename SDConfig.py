#!/usr/bin/env python3

"""
Set global configuration
"""
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# SDK参数
__C.SDCSDK = edict()
# SDK路径
__C.SDCSDK.SDKPATH='selfDrive'
# __C.SDCSDK.SDKPATH='E:\githubCodeSpace\selfDrive'

# 分辨率,图像参数
__C.FRAME = edict()
# 设置图像宽度
__C.FRAME.WIDTH = 512
# 设置图像高度
__C.FRAME.HEIGHT = 256

# 设置鸟瞰图, 摄像头外参数
__C.BIRDVIEW = edict()
# 平视角上边沿缩进长度
__C.BIRDVIEW.upInsideOffset = 35
# 平视角下边沿扩展长度
__C.BIRDVIEW.downOutsideOffset = 100
# 地平线距离屏幕下方长度
__C.BIRDVIEW.horizonUnderHalfHeight = 15

# 车道线模型参数
__C.LANENET = edict()
# 模型路径
__C.LANENET.MODELPATH='models/tusimple_lanenet/tusimple_lanenet_2019-04-10-20-11-49.ckpt-9999'

# yoloV3模型参数
__C.YOLOV3 = edict()

# 模型标签列表
__C.YOLOV3.className='yoloTFModule/coco.names'
# 数据格式 NCHW:GPU only / NHWC
__C.YOLOV3.data_format='NCHW' # Data format: NCHW (gpu only) / NHWC
# 模型路径
__C.YOLOV3.ckpt_file='yoloTFModule/saved_modelTiny/model.ckpt'
# __C.YOLOV3.ckpt_file='yoloTFModule/saved_model/model.ckpt'
# 是否使用tiny
__C.YOLOV3.isTiny=True
# __C.YOLOV3.isTiny=False
# 处理目标检测模型大小
__C.YOLOV3.size=416
# 自信度阈值
__C.YOLOV3.conf_threshold=0.7
# iou阈值
__C.YOLOV3.iou_threshold=0.1
# GPU配置
__C.YOLOV3.gpu_memory_fraction=0.5








