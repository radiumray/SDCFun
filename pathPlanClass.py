
import math
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np

class RobotType(Enum):
    circle = 0
    rectangle = 1

class PPClass:
    def __init__(self, 
                max_accel = 2.0,
                min_accel = -2.0,
                yawRange = 0.5,
                max_speed = 2.0,
                predictTime = 6.0,
                dt = 0.1,
                robotType = RobotType.rectangle,
                robotLength = 1.2,
                robotWidth = 0.5,
                robotRadius = 1.0,
                velocityRate = 0.5,
                yawRate = math.pi / 180.0,
                maxYawRate = 5 * math.pi / 180.0,
                obCostWeight = 6.0,
                distCostWeight = 1.0,
                speedCostWeight = 2.0
    ):
        self.max_accel = max_accel  #最大加速度 [m/ss]
        self.min_accel = min_accel  #最小加速度（允许倒车） [m/ss]
        self.yawRange = yawRange # 偏置范围
        self.max_speed = max_speed # 最大速度
        self.predictTime = predictTime # 预测时间
        self.dt = dt # 时间微分段
        self.robotType = robotType #机器形状
        self.robotLength = robotLength # 机器长
        self.robotWidth = robotWidth # 机器宽
        self.robotRadius = robotRadius # 机器半径
        self.velocityRate = velocityRate # 速度分辨率
        self.yawRate = yawRate # 偏置(转角)分辨率
        self.maxYawRate = maxYawRate # 最大转角范围
        self.obCostWeight = obCostWeight # 障碍成本权重
        self.distCostWeight = distCostWeight # 距离成本权重
        self.speedCostWeight = speedCostWeight # 速度成本权重

    # 更新运动状态模型
    def motion(self, x, u):
        x[0] += u[0] * math.cos(x[2]) * self.dt
        x[1] += u[0] * math.sin(x[2]) * self.dt
        x[2] += u[1] * self.dt
        x[3] = u[0]
        x[4] = u[1]
        return x

    # 绘制方向箭头
    def plot_arrow(self, x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                head_length=width, head_width=width)
        plt.plot(x, y)


    # 预测轨迹
    def predict_trajectory(self, x, v, y):
        x = np.array(x)
        traj = np.array(x)
        time = 0
        while time <= self.predictTime:
            x = self.motion(x, [v, y])
            traj = np.vstack((traj, x))
            time += self.dt
        return traj


    # 计算最终轨迹点到目标点的距离成本
    def calc_to_goal_cost(self, trajectory, goal):
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        dist=np.hypot(dx, dy)
        cost=dist
        return cost

    # 计算障碍物成本信息:碰撞
    def calc_obstacle_cost(self, trajectory, ob):
        ox = ob[:, 0]
        oy = ob[:, 1]
        dx = trajectory[:, 0] - ox[:, None]
        dy = trajectory[:, 1] - oy[:, None]
        r = np.hypot(dx, dy)

        if self.robotType == RobotType.rectangle:
            yaw = trajectory[:, 2]
            rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            rot = np.transpose(rot, [2, 0, 1])
            local_ob = ob[:, None] - trajectory[:, 0:2]
            local_ob = local_ob.reshape(-1, local_ob.shape[-1])
            local_ob = np.array([local_ob @ x for x in rot])
            local_ob = local_ob.reshape(-1, local_ob.shape[-1])
            upper_check = local_ob[:, 0] <= self.robotLength / 2
            right_check = local_ob[:, 1] <= self.robotWidth / 2
            bottom_check = local_ob[:, 0] >= -self.robotLength / 2
            left_check = local_ob[:, 1] >= -self.robotWidth / 2
            if (np.logical_and(np.logical_and(upper_check, right_check),
                            np.logical_and(bottom_check, left_check))).any():
                return float("inf")
        elif self.robotType == RobotType.circle:
            if (r <= self.robotRadius).any():
                return float("inf")

        min_r = np.min(r)
        return 1.0 / min_r  # OK


    # 绘制机器
    def plot_robot(self, x, y, yaw):  # pragma: no cover
        if self.robotType == RobotType.rectangle:
            outline = np.array([[-self.robotLength / 2, self.robotLength / 2,
                                (self.robotLength / 2), -self.robotLength / 2,
                                -self.robotLength / 2],
                                [self.robotWidth / 2, self.robotWidth / 2,
                                - self.robotWidth / 2, -self.robotWidth / 2,
                                self.robotWidth / 2]])
            Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                            [-math.sin(yaw), math.cos(yaw)]])
            outline = (outline.T.dot(Rot1)).T
            outline[0, :] += x
            outline[1, :] += y
            plt.plot(np.array(outline[0, :]).flatten(),
                    np.array(outline[1, :]).flatten(), "-k")
        elif self.robotType == RobotType.circle:
            circle = plt.Circle((x, y), self.robotRadius, color="b")
            plt.gcf().gca().add_artist(circle)
            out_x, out_y = (np.array([x, y]) +
                            np.array([np.cos(yaw), np.sin(yaw)]) * self.robotRadius)
            plt.plot([x, out_x], [y, out_y], "-k")

    # 遍历所有轨迹得出成本最低路径
    def searchBestTrajectory(self, x, ob, goal):
        # 最初成本为正无穷大
        min_cost = float("inf")
        # 遍历所有轨迹得出成本最低路径
        for v in np.arange(self.min_accel, self.max_accel, self.velocityRate):
            for y in np.arange(-self.yawRange, self.yawRange, 4*self.yawRate):
                trajectory = self.predict_trajectory(x, v, y)
                # 计算碰撞风险成本
                ob_cost = self.calc_obstacle_cost(trajectory, ob)
                #如果没有碰撞
                if (ob_cost != float("inf")):
                    # 计算距离成本
                    to_goal_cost = self.calc_to_goal_cost(trajectory, goal)
                    # 计算速度成本
                    speed_cost = (self.max_speed - abs(trajectory[-1, 3]))
                    # 打印轨迹
                    plt.plot(trajectory[:, 0], trajectory[:, 1], ":r", alpha=0.3, linewidth=0.9)
                    # 累加各项成本函数并配置权重
                    final_cost= self.obCostWeight * ob_cost + self.distCostWeight * to_goal_cost + self.speedCostWeight * speed_cost
                else:
                    final_cost=float("inf")

                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, y]
                    best_trajectory = trajectory
        return best_u, best_trajectory            