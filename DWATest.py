import math
import matplotlib.pyplot as plt
import numpy as np
import DWAConfig
from pathPlanClass import PPClass
from pathPlanClass import RobotType

# 获取参数文件
CFG = DWAConfig.cfg

# 初始化路径规划对象
ppClass = PPClass(
    CFG.PATHPLAN.max_accel, #最大加速度 [m/ss]
    CFG.PATHPLAN.min_accel, #最小加速度（允许倒车） [m/ss]
    CFG.PATHPLAN.yawRange, # 偏置范围
    CFG.PATHPLAN.max_speed, # 最大速度
    CFG.PATHPLAN.predictTime, # 预测时间
    CFG.PATHPLAN.dt, # 时间微分段
    CFG.PATHPLAN.robotType, #机器形状
    CFG.PATHPLAN.robotLength, # 机器长
    CFG.PATHPLAN.robotWidth, # 机器宽
    CFG.PATHPLAN.robotRadius, # 机器半径
    CFG.PATHPLAN.velocityRate, # 速度分辨率
    CFG.PATHPLAN.yawRate, # 偏置(转角)分辨率
    CFG.PATHPLAN.maxYawRate, # 最大转角范围
    CFG.PATHPLAN.obCostWeight, # 障碍成本权重
    CFG.PATHPLAN.distCostWeight, # 距离成本权重
    CFG.PATHPLAN.speedCostWeight # 速度成本权重
)

def main():
    # 目的地 [位置x, 位置y]
    goal = np.array(CFG.COORDINATE.goal)
    # 障碍物 [位置x 位置y, ....]
    ob = np.array(CFG.COORDINATE.obstacles)
    # esc键停止模拟
    plt.gcf().canvas.mpl_connect('key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

    # 轨迹初始化起点状态 [位置x, 位置y, 转角, 速度, 角速度]
    robotState=CFG.COORDINATE.robotPoState
    trajectoryHistory = np.array(robotState)

    while True:
        plt.cla()
        # 目标点
        plt.plot(goal[0], goal[1], "xb")
        # 障碍物点
        plt.plot(ob[:, 0], ob[:, 1], "ok")
        # 遍历所有轨迹得出成本最低路径
        best_u, best_trajectory=ppClass.searchBestTrajectory(robotState, ob, goal)
        # 绿线指示预测最好轨迹
        plt.plot(best_trajectory[:, 0], best_trajectory[:, 1], "-g")
        # 更行状态
        robotState = ppClass.motion(robotState, best_u)
        # 存储轨迹状态历史
        trajectoryHistory = np.vstack((trajectoryHistory, robotState)) 
        # 机器当前位置
        plt.plot(robotState[0], robotState[1], "xr")
        # 绘制箭头
        ppClass.plot_arrow(robotState[0], robotState[1], robotState[2], 3 * robotState[3])
        # 绘制机器
        ppClass.plot_robot(robotState[0], robotState[1], robotState[2])
        # 绘制网格
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.0001)
        # 计算是否停止
        dx = goal[0] - robotState[0]
        dy = goal[1] - robotState[1]
        dist=np.hypot(dx, dy)
        if(dist<1):
            break
    # 绘制历史轨迹
    plt.plot(trajectoryHistory[:, 0], trajectoryHistory[:, 1], "-g")
    plt.pause(0.0001)
    plt.show()

if __name__ == '__main__':
    main()