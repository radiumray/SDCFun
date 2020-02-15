#!/usr/bin/env python3

"""
Set global configuration
"""
from easydict import EasyDict as edict
import math
from pathPlanClass import RobotType

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# SDK参数
__C.DWASDK = edict()
# SDK路径
__C.DWASDK.SDKPATH='DWALib'
# __C.DWASDK.SDKPATH='E:\githubCodeSpace\selfDrive'


# 分辨率,图像参数
__C.FRAME = edict()
# 设置图像宽度
__C.FRAME.WIDTH = 500
# 设置图像高度
__C.FRAME.HEIGHT = 500



# 设置初始化坐标
__C.COORDINATE = edict()
# 机器初始位置
__C.COORDINATE.robotPoState = [0.0, 0.0, math.pi / 8.0, 0.0, 0.0]
# 平视角下边沿扩展长度
__C.COORDINATE.goal = [10, 10]
# 地平线距离屏幕下方长度
__C.COORDINATE.obstacles = [[-10, -10],
                            [0, 2],
                            [4.0, 2.0],
                            [5.0, 4.0],
                            [5.0, 5.0],
                            [5.0, 6.0],
                            [5.0, 9.0],
                            [8.0, 9.0],
                            [7.0, 9.0],
                            [8.0, 10.0],
                            [9.0, 11.0],
                            [12.0, 13.0],
                            [12.0, 12.0],
                            [15.0, 15.0],
                            [13.0, 13.0]
                            ]


# 设置路径规划参数
__C.PATHPLAN = edict()
__C.PATHPLAN.max_accel = 2.0  # 最大加速度 [m/ss]
__C.PATHPLAN.min_accel = -2.0  # 最小加速度（允许倒车） [m/ss]
__C.PATHPLAN.yawRange = 0.5 # 偏置范围
__C.PATHPLAN.max_speed = 2.0 # 最大速度
__C.PATHPLAN.predictTime=6.0 # 预测时间
__C.PATHPLAN.dt=0.1 # 时间微分段
__C.PATHPLAN.robotType=RobotType.rectangle #机器形状
__C.PATHPLAN.robotLength=1.2 # 机器长
__C.PATHPLAN.robotWidth=0.5 # 机器宽
__C.PATHPLAN.robotRadius=1.0 # 机器半径
__C.PATHPLAN.velocityRate=0.5 # 速度分辨率
__C.PATHPLAN.yawRate=math.pi / 180.0 # 偏置(转角)分辨率
__C.PATHPLAN.maxYawRate=5 * __C.PATHPLAN.yawRate # 最大转角范围
__C.PATHPLAN.obCostWeight=6.0 # 障碍成本权重
__C.PATHPLAN.distCostWeight=1.0 # 距离成本权重
__C.PATHPLAN.speedCostWeight=2.0 # 速度成本权重