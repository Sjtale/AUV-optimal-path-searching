import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
from gym.spaces import Discrete, Box
from gym import Env
import matplotlib.pyplot as plt
import random
import copy
import imageio

# 检查是否有可用的GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class Auv_SimpleAction(Env):
    """
    简单的环境：
    动作个数: 6 向上浮，向下潜,前、后、左、右
    奖励机制简单，燃油消耗即时间消耗
    """
    def __init__(self, max_fuels, image_save_path, changeable = False) -> None:
        # 动作个数：
        self.env_name = 'Simple'
        self.action_space = Discrete(6)
        self.observation_space = Box(low=0, high=10, shape=(3,), dtype=np.float32)
        self.changeable = changeable
        self.image_save_path = image_save_path

        #初始化障碍物
        self.obstacles = [
            {'x': 4, 'y': 6, 'z': 0, 'radius': 1.5},
            {'x': 5, 'y': 1, 'z': 0, 'radius': 1.0},
            {'x': 1, 'y': 5, 'z': 0, 'radius': 1.2},
            {'x': 8, 'y': 0, 'z': 0, 'radius': 0.8},
            {'x': 1, 'y': 3, 'z': 3, 'radius': 0.8},
        ]

        if changeable:
            random.seed(123)
            valid_positions = False
            
            while not valid_positions:
                # 随机初始化起点
                self.agent_x = random.randint(0, 10)
                self.agent_y = random.randint(0, 10)
                self.agent_z = random.randint(0, 10)

                # 随机初始化终点
                self.goal_x = random.randint(0, 10)
                self.goal_y = random.randint(0, 10)
                self.goal_z = random.randint(0, 10)

                # 起点和重点不重合
                if self.agent_x == self.goal_x and self.agent_y == self.goal_y and self.agent_z == self.goal_z:
                    continue

                #起点和终点不能在障碍物里面
                valid_positions = True
                for obstacle in self.obstacles:
                    dist1 = np.sqrt((self.agent_x - obstacle['x'])**2 + 
                                    (self.agent_y - obstacle['y'])**2 + 
                                    (self.agent_z - obstacle['z'])**2)
                    dist2 = np.sqrt((self.goal_x - obstacle['x'])**2 + 
                                    (self.goal_y - obstacle['y'])**2 + 
                                    (self.goal_z - obstacle['z'])**2)
                    if dist1 <= obstacle['radius'] or dist2 <= obstacle['radius']:
                        valid_positions = False
                        break  
        else:             
            # 初始化起点 [0,0,0]
            self.agent_x = 0
            self.agent_y = 0
            self.agent_z = 0

            #初始化终点 [10,10,10]
            self.goal_x = 10
            self.goal_y = 10
            self.goal_z = 10

        self.danger = 0
        self.fuelcost = 0
        
        self.randomize_water_current()
        self.max_fuels = max_fuels
        self.current_time = 0
        self.trajectory = []

        self.trace_trajectory = []
        self.trace_danger = []
        self.trace_time = []
        self.trace_fuelcost = []

    def randomize_water_current(self):
        self.water_current_x = -0.05
        self.water_current_y = 0
        self.water_current_z = 0.05

    def step(self, action):

        self.current_time += 1
        self.fuelcost += 1

        oridist_between_st_end = np.sqrt((self.agent_x - self.goal_x)**2 + 
                           (self.agent_y - self.goal_y)**2 + 
                           (self.agent_z - self.goal_z)**2)
        
        if action == 0: 
            # 上升
            self.agent_z += 1
            self.fuelcost += 1
        elif action == 1: 
            # 下降
            self.agent_z -= 1
            self.fuelcost += 1
        elif action == 2: 
            self.agent_x += 1
        elif action == 3: 
            self.agent_x -= 1
        elif action == 4: 
            self.agent_y += 1
        elif action == 5: 
            self.agent_y -= 1

        
        # 加上水流的影响
        self.agent_x += self.water_current_x
        self.agent_y += self.water_current_y
        self.agent_z += self.water_current_z

        done = False

        self.agent_x = min(max(0, self.agent_x), 10)
        self.agent_y = min(max(0, self.agent_y), 10)
        self.agent_z = min(max(0, self.agent_z), 10)
        self.trajectory.append((self.agent_x, self.agent_y, self.agent_z))
        
        nowdist_between_st_end = np.sqrt((self.agent_x - self.goal_x)**2 + 
                           (self.agent_y - self.goal_y)**2 + 
                           (self.agent_z - self.goal_z)**2)
        
        reward = -1
        # if nowdist_between_st_end < oridist_between_st_end:
        #     # 离终点近了
        #     reward = 0
        # else:
        #     reward = -1

        for obstacle in self.obstacles:
            # 碰到障碍物则终止
            dist = np.sqrt((self.agent_x - obstacle['x'])**2 + 
                           (self.agent_y - obstacle['y'])**2 + 
                           (self.agent_z - obstacle['z'])**2)
            
            self.danger += 10/ dist
            if dist <= obstacle['radius']:
                done = True
                reward = -2000
        
        #距离出口距离等于 0 判断到达出口
        if nowdist_between_st_end == 0:
            # 到达终点终止
            done = True
            reward = 2000
            print("Success")

        if self.fuelcost >= self.max_fuels:
            # 如果序列长度过长则直接终止
            done = True
            reward = -2000
    

        observation = np.array([self.agent_x, self.agent_y, self.agent_z], dtype=np.float32)
        
        return observation, reward, done

    def reset(self):
        self.agent_x = 0
        self.agent_y = 0
        self.agent_z = 0
        
        self.trace_trajectory.append(self.trajectory)  # Store the completed episode's trajectory
        self.trajectory = [(self.agent_x, self.agent_y, self.agent_z)]

        self.trace_danger.append(self.danger)
        self.trace_time.append(self.current_time)
        self.trace_fuelcost.append(self.fuelcost)

        self.danger = 0
        self.current_time = 0
        self.fuelcost = 0

        observation = np.array([self.agent_x, self.agent_y, self.agent_z], dtype=np.float32)
        return observation

    def render(self):
        """
        可视化一个序列的路径
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        trajectory = np.array(self.trajectory)
        num_points = len(trajectory)
        colors = np.linspace(0.2, 1, num_points)  # 范围从浅色 (0.2) 到深色 (1)

        for i in range(num_points - 1):
            ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2], color=(0, 0, colors[i]), alpha=0.7)

        ax.scatter(0, 0, 0, color='green', label='Start')
        ax.scatter(self.goal_x, self.goal_y, self.goal_z, color='red', label='Goal')
        
        # 绘制障碍物
        for obstacle in self.obstacles:
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = obstacle['radius'] * np.outer(np.cos(u), np.sin(v)) + obstacle['x']
            y = obstacle['radius'] * np.outer(np.sin(u), np.sin(v)) + obstacle['y']
            z = obstacle['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle['z']
            ax.plot_surface(x, y, z, color='black', alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        plt.savefig(self.image_save_path + '.png', format='png', dpi=800)
        plt.show()
    


    def visualize_all_trajectories(self):
        """
        可视化所有走过的点
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for traj in self.trace_trajectory:
            trajectory = np.array(traj)
            num_points = len(trajectory)
            colors = np.linspace(0.2, 1, num_points)

            for i in range(num_points - 1):
                ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2], color=(0, 0, colors[i]), alpha=0.7)

        ax.scatter(3, 3, 3, color='green', label='Start')
        ax.scatter(self.goal_x, self.goal_y, self.goal_z, color='red', label='Goal')

        for obstacle in self.obstacles:
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = obstacle['radius'] * np.outer(np.cos(u), np.sin(v)) + obstacle['x']
            y = obstacle['radius'] * np.outer(np.sin(u), np.sin(v)) + obstacle['y']
            z = obstacle['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle['z']
            ax.plot_surface(x, y, z, color='black', alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title('All Agent Trajectories')
        plt.show()

    def visualize_metrics(self):
        """
        可视化 时间、燃油量、危险系数 变化
        """
        
        epochs = range(len(self.trace_time))
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        # Time plot
        axs[0].plot(epochs, self.trace_time, label='Time', color='blue')
        axs[0].set_title('Time per Epoch')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Time')
        axs[0].legend()
        
        # fuel plot
        axs[1].plot(epochs, self.trace_fuelcost, label='Fuel', color='Yellow')
        axs[1].set_title('Time per Epoch')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('fuel')
        axs[1].legend()

        # Danger plot
        axs[2].plot(epochs, self.trace_danger, label='Danger', color='red')
        axs[2].set_title('Danger per Epoch')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Danger')
        axs[2].legend()
        
        plt.tight_layout()
        plt.savefig(self.image_save_path + '-metrics' + '.png', format='png', dpi=300)
        plt.show()

class Auv_MultiActions(Env):
    """
    较为完善的环境
    动作空间 13: 前、后，上浮，下潜 以及 9 个角度
    燃油消耗机制： 转向角度越大，燃油消耗越多
    时间消耗为每一个 step 加一
    """
    def __init__(self, max_fuel, image_save_path) -> None:
        # 动作个数：向上浮，向下潜,向前，向后 修改角度
        self.action_space = Discrete(12)
        self.observation_space = Box(low=0, high=10, shape=(3,), dtype=np.float32)

        self.image_save_path = image_save_path
        # 初始化起点 [0,0,0]
        self.agent_x = 0
        self.agent_y = 0
        self.agent_z = 0

        #初始化终点 [10,10,10]
        self.goal_x = 10
        self.goal_y = 10
        self.goal_z = 10

        # 初始化朝向
        self.angle = 0

        #初始化障碍物
        self.obstacles = [
            {'x': 4, 'y': 6, 'z': 0, 'radius': 1.5},
            {'x': 5, 'y': 1, 'z': 0, 'radius': 1.0},
            {'x': 1, 'y': 5, 'z': 0, 'radius': 1.2},
            {'x': 8, 'y': 0, 'z': 0, 'radius': 0.8},
            {'x': 1, 'y': 3, 'z': 3, 'radius': 0.8},
        ]

        np.random.seed(123)
        self.randomize_water_current()
        self.max_fuel = max_fuel # AUV 最大燃油

        self.current_fuel = 0 # 当前燃油消耗量
        self.time = 0
        self.danger = 0
        

        self.trajectory = [] # AUV走过的路劲集合

        self.trace_trajectory = [] # 所有epoch AUV走过的路劲集合
        self.trace_fuel = [] # 所有epoch AUV走过的一次路劲的燃油消耗总量
        self.trace_danger = [] # 所有epoch AUV走过的一次路劲的危险系数之和
        self.trace_time = [] # 所有epoch AUV走过的一次路劲所消耗的时间

    def randomize_water_current(self):
        self.water_current_x = 0
        self.water_current_y = 0
        self.water_current_z = 0
        # self.water_current_x = -0.2
        # self.water_current_y = -0.2
        # self.water_current_z = 0.2

    def step(self, action):

        self.current_fuel += 1
        self.time += 1
        new_angle = self.angle

        oridist_between_st_end = np.sqrt((self.agent_x - self.goal_x)**2 + 
                           (self.agent_y - self.goal_y)**2 + 
                           (self.agent_z - self.goal_z)**2)
        
        if action == 0: 
            # 上升
            self.agent_z += 1
        elif action == 1: 
            # 下降
            self.agent_z -= 1
        elif action == 2: 
            # 前进
            self.agent_x += np.cos(np.radians(self.angle)) * 2
            self.agent_y += np.sin(np.radians(self.angle)) * 2
        elif action == 3: 
            # 后退
            self.agent_x -= np.cos(np.radians(self.angle)) * 2
            self.agent_y -= np.sin(np.radians(self.angle)) * 2
        elif action == 4: 
            new_angle = 45
        elif action == 5: 
            new_angle = 90
        elif action == 6: 
            new_angle = 135
        elif action == 7: 
            new_angle = 180
        elif action == 8: 
            new_angle = 225
        elif action == 9: 
            new_angle = 270
        elif action == 10: 
            new_angle = 315
        elif action == 11: 
            new_angle = 0


        done = False

        self.current_fuel += abs(new_angle - self.angle) / 72 ## 转向需要更多燃油

        self.angle = new_angle
        
        # 加上水流的影响
        self.agent_x += self.water_current_x
        self.agent_y += self.water_current_y
        self.agent_z += self.water_current_z

        self.agent_x = min(max(0, self.agent_x), 10)
        self.agent_y = min(max(0, self.agent_y), 10)
        self.agent_z = min(max(0, self.agent_z), 10)

        self.trajectory.append((self.agent_x, self.agent_y, self.agent_z))
        nowdist_between_st_end = np.sqrt((self.agent_x - self.goal_x)**2 + 
                           (self.agent_y - self.goal_y)**2 + 
                           (self.agent_z - self.goal_z)**2)
        reward = -1
        # if nowdist_between_st_end <= oridist_between_st_end:
        #     # 离终点近了
        #     reward = 0
        # else:
        #     reward = -1

        for obstacle in self.obstacles:
            # 碰到障碍物则终止
            dist = np.sqrt((self.agent_x - obstacle['x'])**2 + 
                           (self.agent_y - obstacle['y'])**2 + 
                           (self.agent_z - obstacle['z'])**2)
            self.danger += dist # 更新danger系数
            if dist <= obstacle['radius']:
                done = True
                reward = -2000

        exitdist = np.sqrt((self.agent_x - 10)**2 + 
                           (self.agent_y - 10)**2 + 
                           (self.agent_z - 10)**2)
        if exitdist < 1:
            # 到达终点终止
            done = True
            reward = 2000
            print("Success")

        if self.current_fuel >= self.max_fuel:
            # 燃油耗尽直接终止
            done = True
            reward = -2000
        

        observation = np.array([self.agent_x, self.agent_y, self.agent_z], dtype=np.float32)
        
        return observation, reward, done

    def reset(self):
        self.agent_x = 0
        self.agent_y = 0
        self.agent_z = 0

        self.angle = 0

        self.trace_trajectory.append(self.trajectory)  # 保存当前一个完整序列所走过的路径
        self.trace_danger.append(self.danger)
        self.trace_fuel.append(self.current_fuel)
        self.trace_time.append(self.time)

        self.trajectory = [(self.agent_x, self.agent_y, self.agent_z)]
        self.current_fuel = 0 # 初始化当前消耗油为 0 
        self.danger = 0
        self.time = 0

        observation = np.array([self.agent_x, self.agent_y, self.agent_z], dtype=np.float32)
        return observation

    def render(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        trajectory = np.array(self.trajectory)
        num_points = len(trajectory)
        colors = np.linspace(0.2, 1, num_points)  # 范围从浅色 (0.2) 到深色 (1)

        for i in range(num_points - 1):
            ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2], color=(0, 0, colors[i]), alpha=0.7)

        ax.scatter(0, 0, 0, color='green', label='Start')
        ax.scatter(self.goal_x, self.goal_y, self.goal_z, color='red', label='Goal')
        
        # 绘制障碍物
        for obstacle in self.obstacles:
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = obstacle['radius'] * np.outer(np.cos(u), np.sin(v)) + obstacle['x']
            y = obstacle['radius'] * np.outer(np.sin(u), np.sin(v)) + obstacle['y']
            z = obstacle['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle['z']
            ax.plot_surface(x, y, z, color='black', alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        plt.savefig(self.image_save_path + '.png', format='png', dpi=800)
        plt.show()

    def visualize_all_trajectories(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for traj in self.trace_trajectory:
            trajectory = np.array(traj)
            num_points = len(trajectory)
            colors = np.linspace(0.2, 1, num_points)

            for i in range(num_points - 1):
                ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2], color=(0, 0, colors[i]), alpha=0.7)

        ax.scatter(3, 3, 3, color='green', label='Start')
        ax.scatter(self.goal_x, self.goal_y, self.goal_z, color='red', label='Goal')

        for obstacle in self.obstacles:
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = obstacle['radius'] * np.outer(np.cos(u), np.sin(v)) + obstacle['x']
            y = obstacle['radius'] * np.outer(np.sin(u), np.sin(v)) + obstacle['y']
            z = obstacle['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle['z']
            ax.plot_surface(x, y, z, color='black', alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title('All Agent Trajectories')
        plt.show()

    def visualize_metrics(self):
        epochs = range(len(self.trace_time))
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        # Time plot
        axs[0].plot(epochs, self.trace_time, label='Time', color='blue')
        axs[0].set_title('Time per Epoch')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Time')
        axs[0].legend()
        
        # Fuel plot
        axs[1].plot(epochs, self.trace_fuel, label='Fuel', color='green')
        axs[1].set_title('Fuel Consumption per Epoch')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Fuel')
        axs[1].legend()
        
        # Danger plot
        axs[2].plot(epochs, self.trace_danger, label='Danger', color='red')
        axs[2].set_title('Danger per Epoch')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Danger')
        axs[2].legend()
        
        plt.tight_layout()
        plt.savefig(self.image_save_path + '--metrics' + '.png', format='png', dpi=800)
        plt.show()


class Auv_changeable(Env):
    """
    起点终点不固定的环境：
    动作个数: 6 向上浮，向下潜,前、后、左、右
    奖励机制简单，燃油消耗即时间消耗
    """
    def __init__(self, max_fuels, image_save_path, changeable = True) -> None:
        # 动作个数：
        self.env_name = 'Simple'
        self.action_space = Discrete(6)
        self.observation_space = Box(low=0, high=10, shape=(9,), dtype=np.float32)
        self.changeable = changeable
        self.image_save_path = image_save_path

        #初始化障碍物
        self.obstacles = [
            {'x': 4, 'y': 6, 'z': 0, 'radius': 1.5},
            {'x': 5, 'y': 1, 'z': 0, 'radius': 1.0},
            {'x': 1, 'y': 5, 'z': 0, 'radius': 1.2},
            {'x': 8, 'y': 0, 'z': 0, 'radius': 0.8},
            {'x': 1, 'y': 3, 'z': 3, 'radius': 0.8},
        ]

        # 初始化起点终点
        self.randomize_start_end()

        

        self.danger = 0 #危险系数
        self.fuelcost = 0 # 燃油量
        
        self.randomize_water_current() # 初始化水流方向
        self.max_fuels = max_fuels # 最大燃油量
        self.current_time = 0 # 当前时间
        self.trajectory = []

        self.trace_trajectory = []
        self.trace_danger = []
        self.trace_time = []
        self.trace_fuelcost = []

    def randomize_water_current(self):
        self.water_current_x = 0
        self.water_current_y = 0
        self.water_current_z = 0

    def randomize_start_end(self):
        valid_positions = False
        
        while not valid_positions:
            # 随机初始化起点
            self.begin_x = random.randint(0, 10)
            self.begin_y = random.randint(0, 10)
            self.begin_z = random.randint(0, 10)

            # 随机初始化终点
            self.goal_x = random.randint(0, 10)
            self.goal_y = random.randint(0, 10)
            self.goal_z = random.randint(0, 10)

            # 起点和重点不重合
            if self.begin_x == self.goal_x and self.begin_y == self.goal_y and self.begin_z == self.goal_z:
                continue

            #起点和终点不能在障碍物里面
            valid_positions = True
            for obstacle in self.obstacles:
                dist1 = np.sqrt((self.begin_x - obstacle['x'])**2 + 
                                (self.begin_y - obstacle['y'])**2 + 
                                (self.begin_z - obstacle['z'])**2)
                dist2 = np.sqrt((self.goal_x - obstacle['x'])**2 + 
                                (self.goal_y - obstacle['y'])**2 + 
                                (self.goal_z - obstacle['z'])**2)
                if dist1 <= obstacle['radius'] or dist2 <= obstacle['radius']:
                    valid_positions = False
                    break
        #初始化 agent 当前的位置
        self.agent_x = copy.deepcopy(self.begin_x)
        self.agent_y = copy.deepcopy(self.begin_y)
        self.agent_z = copy.deepcopy(self.begin_z)


    def step(self, action):

        self.current_time += 1
        self.fuelcost += 1
        reward = -1
        
        oridist_between_st_end = np.sqrt((self.agent_x - self.goal_x)**2 + 
                           (self.agent_y - self.goal_y)**2 + 
                           (self.agent_z - self.goal_z)**2)
        
        if action == 0: 
            # 上升
            self.agent_z += 1
        elif action == 1: 
            # 下降
            self.agent_z -= 1
        elif action == 2: 
            self.agent_x += 1
        elif action == 3: 
            self.agent_x -= 1
        elif action == 4: 
            self.agent_y += 1
        elif action == 5: 
            self.agent_y -= 1

        
        # 加上水流的影响
        self.agent_x += self.water_current_x
        self.agent_y += self.water_current_y
        self.agent_z += self.water_current_z

        done = False

        self.agent_x = min(max(0, self.agent_x), 10)
        self.agent_y = min(max(0, self.agent_y), 10)
        self.agent_z = min(max(0, self.agent_z), 10)
        self.trajectory.append([self.agent_x, self.agent_y, self.agent_z,self.begin_x, self.begin_y, self.begin_z, self.goal_x,self.goal_y,self.goal_z])
        
        nowdist_between_st_end = np.sqrt((self.agent_x - self.goal_x)**2 + 
                           (self.agent_y - self.goal_y)**2 + 
                           (self.agent_z - self.goal_z)**2)
        
        if nowdist_between_st_end < oridist_between_st_end:
            # 离终点近了
            reward = 0
        else:
            reward = -1
        # reward = -1
        for obstacle in self.obstacles:
            # 碰到障碍物则终止
            dist = np.sqrt((self.agent_x - obstacle['x'])**2 + 
                           (self.agent_y - obstacle['y'])**2 + 
                           (self.agent_z - obstacle['z'])**2)
            
            self.danger += 10/ (dist + 0.01)
            if dist <= obstacle['radius']:
                done = True
                reward = -50
                break
        
        #距离出口距离小于 1 判断到达出口
        if nowdist_between_st_end <= 0:
            # 到达终点终止
            done = True
            reward = 100
            print("Success")

        if self.fuelcost >= self.max_fuels:
            # 如果序列长度过长则直接终止
            done = True
            reward = -20
    
        
        observation = np.array([self.agent_x, self.agent_y, self.agent_z,self.begin_x, self.begin_y, self.begin_z, self.goal_x,self.goal_y,self.goal_z], dtype=np.float32)
        
        return observation, reward, done

    def reset(self):
        
        self.trace_trajectory.append(self.trajectory)  # Store the completed episode's trajectory
        self.randomize_start_end()
        self.trajectory = [(self.agent_x, self.agent_y, self.agent_z,self.begin_x, self.begin_y, self.begin_z, self.goal_x,self.goal_y,self.goal_z)]

        self.trace_danger.append(self.danger)
        self.trace_time.append(self.current_time)
        self.trace_fuelcost.append(self.fuelcost)

        self.danger = 0
        self.current_time = 0
        self.fuelcost = 0
        

        observation = np.array([self.agent_x, self.agent_y, self.agent_z,self.begin_x, self.begin_y, self.begin_z, self.goal_x,self.goal_y,self.goal_z], dtype=np.float32)
        return observation

    def get_next_state(self, current_state, action):
        x, y, z = current_state
        if action == 0: 
            # 上升
            z += 1
        elif action == 1: 
            # 下降
            z -= 1
        elif action == 2: 
            x += 1
        elif action == 3: 
            x -= 1
        elif action == 4: 
            y += 1
        elif action == 5: 
            y -= 1
        
        done = False

        x = min(max(0, x), 10)
        y = min(max(0, y), 10)
        z = min(max(0, z), 10)

        for obstacle in self.obstacles:
            # 碰到障碍物则终止
            dist = np.sqrt((x - obstacle['x'])**2 + 
                           (y - obstacle['y'])**2 + 
                           (z - obstacle['z'])**2)
            
            if dist <= obstacle['radius']:
                done = True
                break
        return (x, y, z), done

    def render(self):
        """
        可视化一个序列的路径
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        trajectory = np.array(self.trajectory)
        num_points = len(trajectory)
        colors = np.linspace(0.2, 1, num_points)  # 范围从浅色 (0.2) 到深色 (1)

        for i in range(num_points - 1):
            ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2], color=(0, 0, colors[i]), alpha=0.7)

        ax.scatter(self.begin_x, self.begin_y, self.begin_z, color='green', label='Start')
        ax.scatter(self.goal_x, self.goal_y, self.goal_z, color='red', label='Goal')
        
        # 绘制障碍物
        for obstacle in self.obstacles:
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = obstacle['radius'] * np.outer(np.cos(u), np.sin(v)) + obstacle['x']
            y = obstacle['radius'] * np.outer(np.sin(u), np.sin(v)) + obstacle['y']
            z = obstacle['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle['z']
            ax.plot_surface(x, y, z, color='black', alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        plt.savefig(self.image_save_path + '.png', format='png', dpi=800)
        plt.show()
        
    
    def render_gif(self):
        """
        可视化一个序列的路径并生成GIF
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        trajectory = np.array(self.trajectory)
        num_points = len(trajectory)
        colors = np.linspace(0.2, 1, num_points)  # 范围从浅色 (0.2) 到深色 (1)

        images = []  # 存储每个帧的路径

        for i in range(num_points):
            ax.clear()  # 清除当前图像
            
            # 画轨迹
            if i > 0:
                ax.plot(trajectory[:i+1, 0], trajectory[:i+1, 1], trajectory[:i+1, 2], color='blue', alpha=0.7)

            # 当前点
            ax.scatter(trajectory[i, 0], trajectory[i, 1], trajectory[i, 2], color='blue')

            # 绘制起点和终点
            ax.scatter(self.begin_x, self.begin_y, self.begin_z, color='green', label='Start')
            ax.scatter(self.goal_x, self.goal_y, self.goal_z, color='red', label='Goal')

            # 绘制障碍物
            for obstacle in self.obstacles:
                u = np.linspace(0, 2 * np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                x = obstacle['radius'] * np.outer(np.cos(u), np.sin(v)) + obstacle['x']
                y = obstacle['radius'] * np.outer(np.sin(u), np.sin(v)) + obstacle['y']
                z = obstacle['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle['z']
                ax.plot_surface(x, y, z, color='black', alpha=0.5)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()

            # 保存每个帧到图像
            temp_image_path = f"{self.image_save_path}_frame_{i}.png"
            plt.savefig(temp_image_path, format='png', dpi=100)
            images.append(imageio.imread(temp_image_path))

        # 生成GIF
        gif_path = self.image_save_path + '.gif'
        imageio.mimsave(gif_path, images, duration=0.5)  # duration为每帧显示的时间
        plt.show()




    def visualize_all_trajectories(self):
        """
        可视化所有走过的点
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for traj in self.trace_trajectory:
            trajectory = np.array(traj)
            num_points = len(trajectory)
            colors = np.linspace(0.2, 1, num_points)

            for i in range(num_points - 1):
                ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2], color=(0, 0, colors[i]), alpha=0.7)

        ax.scatter(3, 3, 3, color='green', label='Start')
        ax.scatter(self.goal_x, self.goal_y, self.goal_z, color='red', label='Goal')

        for obstacle in self.obstacles:
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = obstacle['radius'] * np.outer(np.cos(u), np.sin(v)) + obstacle['x']
            y = obstacle['radius'] * np.outer(np.sin(u), np.sin(v)) + obstacle['y']
            z = obstacle['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle['z']
            ax.plot_surface(x, y, z, color='black', alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title('All Agent Trajectories')
        plt.show()

    def visualize_metrics(self):
        """
        可视化 时间、燃油量、危险系数 变化
        """
        
        epochs = range(len(self.trace_time))
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        # Time plot
        axs[0].plot(epochs, self.trace_time, label='Time', color='blue')
        axs[0].set_title('Time per Epoch')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Time')
        axs[0].legend()
        
        # fuel plot
        axs[1].plot(epochs, self.trace_fuelcost, label='Fuel', color='Yellow')
        axs[1].set_title('Time per Epoch')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('fuel')
        axs[1].legend()

        # Danger plot
        axs[2].plot(epochs, self.trace_danger, label='Danger', color='red')
        axs[2].set_title('Danger per Epoch')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Danger')
        axs[2].legend()
        
        plt.tight_layout()
        plt.savefig(self.image_save_path + '-metrics' + '.png', format='png', dpi=300)
        plt.show()



class Auv_blined(Env):
    """
    限制视野的环境：
    动作个数: 6 向上浮，向下潜,前、后、左、右
    奖励机制简单，燃油消耗即时间消耗
    """
    def __init__(self, max_fuels, image_save_path, vision_area = 3) -> None:
        # 动作个数：
        self.env_name = 'Simple'
        self.action_space = Discrete(6)
        self.observation_space = Box(low=0, high=10, shape=(30,), dtype=np.float32)
        self.vision_area = vision_area
        self.image_save_path = image_save_path

        #初始化障碍物
        self.obstacles = [
            {'x': 4, 'y': 6, 'z': 0, 'radius': 1.5},
            {'x': 5, 'y': 1, 'z': 0, 'radius': 1.0},
            {'x': 1, 'y': 5, 'z': 0, 'radius': 1.2},
            {'x': 8, 'y': 0, 'z': 0, 'radius': 0.8},
            {'x': 1, 'y': 3, 'z': 3, 'radius': 0.8},
        ]

        # 初始化起点 [0,0,0]

        self.agent_x = 0
        self.agent_y = 0
        self.agent_z = 0

        #初始化终点 [10,10,10]
        self.goal_x = 10
        self.goal_y = 10
        self.goal_z = 10

        self.danger = 0
        self.fuelcost = 0
        
        self.randomize_water_current()
        self.max_fuels = max_fuels
        self.current_time = 0
        self.trajectory = []

        self.trace_trajectory = []
        self.trace_danger = []
        self.trace_time = []
        self.trace_fuelcost = []


    def randomize_water_current(self):
        self.water_current_x = -0.05
        self.water_current_y = 0
        self.water_current_z = 0.05

    def get_surround_info(self):
        x = self.agent_x
        y = self.agent_y
        z = self.agent_z
        info = []
        for i in range(3):
            x = min(max(0, x + i), 10)
            for j in range(3):
                y =  min(max(0, y + j), 10)
                for k in range(3):
                    z = min(max(0, z + k), 10)
                    flag = 0
                    for obstacle in self.obstacles:
                        # 碰到障碍物则终止
                        dist = np.sqrt((x - obstacle['x'])**2 + 
                                        (y - obstacle['y'])**2 + 
                                        (z - obstacle['z'])**2)
                        if dist <= obstacle['radius']:
                            info.append(-1)
                            flag = 1
                            break
                    if flag == 0:
                        info.append(1)
        return info
                        

    def step(self, action):

        self.current_time += 1
        self.fuelcost += 1

        oridist_between_st_end = np.sqrt((self.agent_x - self.goal_x)**2 + 
                           (self.agent_y - self.goal_y)**2 + 
                           (self.agent_z - self.goal_z)**2)
        
        if action == 0: 
            # 上升
            self.agent_z += 1
        elif action == 1: 
            # 下降
            self.agent_z -= 1
        elif action == 2: 
            self.agent_x += 1
        elif action == 3: 
            self.agent_x -= 1
        elif action == 4: 
            self.agent_y += 1
        elif action == 5: 
            self.agent_y -= 1

        
        # 加上水流的影响
        self.agent_x += self.water_current_x
        self.agent_y += self.water_current_y
        self.agent_z += self.water_current_z

        done = False

        self.agent_x = min(max(0, self.agent_x), 10)
        self.agent_y = min(max(0, self.agent_y), 10)
        self.agent_z = min(max(0, self.agent_z), 10)
        self.trajectory.append((self.agent_x, self.agent_y, self.agent_z))
        
        nowdist_between_st_end = np.sqrt((self.agent_x - self.goal_x)**2 + 
                           (self.agent_y - self.goal_y)**2 + 
                           (self.agent_z - self.goal_z)**2)
        

        if nowdist_between_st_end <= oridist_between_st_end:
            # 离终点近了
            reward = 1
        else:
            reward = -1

        for obstacle in self.obstacles:
            # 碰到障碍物则终止
            dist = np.sqrt((self.agent_x - obstacle['x'])**2 + 
                           (self.agent_y - obstacle['y'])**2 + 
                           (self.agent_z - obstacle['z'])**2)
            
            self.danger += 10/ dist
            if dist <= obstacle['radius']:
                done = True
                reward = -2000
        
        #距离出口距离小于 1 判断到达出口
        if nowdist_between_st_end < 1:
            # 到达终点终止
            done = True
            reward = 2000
            print("Success")

        if self.fuelcost >= self.max_fuels:
            # 如果序列长度过长则直接终止
            done = True
            reward = -2000
        info = self.get_surround_info()
        info.extend([self.agent_x, self.agent_y, self.agent_z])
        
        observation = np.array(info, dtype=np.float32)
        
        return observation, reward, done

    def reset(self):
        self.agent_x = 0
        self.agent_y = 0
        self.agent_z = 0
        
        self.trace_trajectory.append(self.trajectory)  # Store the completed episode's trajectory
        self.trajectory = [(self.agent_x, self.agent_y, self.agent_z)]

        self.trace_danger.append(self.danger)
        self.trace_time.append(self.current_time)
        self.trace_fuelcost.append(self.fuelcost)

        self.danger = 0
        self.current_time = 0
        self.fuelcost = 0
        info = self.get_surround_info()
        info.extend([self.agent_x, self.agent_y, self.agent_z])
        
        observation = np.array(info, dtype=np.float32)
        return observation

    def render(self):
        """
        可视化一个序列的路径
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        trajectory = np.array(self.trajectory)
        num_points = len(trajectory)
        colors = np.linspace(0.2, 1, num_points)  # 范围从浅色 (0.2) 到深色 (1)

        for i in range(num_points - 1):
            ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2], color=(0, 0, colors[i]), alpha=0.7)

        ax.scatter(0, 0, 0, color='green', label='Start')
        ax.scatter(self.goal_x, self.goal_y, self.goal_z, color='red', label='Goal')
        
        # 绘制障碍物
        for obstacle in self.obstacles:
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = obstacle['radius'] * np.outer(np.cos(u), np.sin(v)) + obstacle['x']
            y = obstacle['radius'] * np.outer(np.sin(u), np.sin(v)) + obstacle['y']
            z = obstacle['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle['z']
            ax.plot_surface(x, y, z, color='black', alpha=0.5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        plt.savefig(self.image_save_path + '.png', format='png', dpi=800)
        plt.show()
        

    def visualize_all_trajectories(self):
        """
        可视化所有走过的点
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for traj in self.trace_trajectory:
            trajectory = np.array(traj)
            num_points = len(trajectory)
            colors = np.linspace(0.2, 1, num_points)

            for i in range(num_points - 1):
                ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], trajectory[i:i+2, 2], color=(0, 0, colors[i]), alpha=0.7)

        ax.scatter(3, 3, 3, color='green', label='Start')
        ax.scatter(self.goal_x, self.goal_y, self.goal_z, color='red', label='Goal')

        for obstacle in self.obstacles:
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = obstacle['radius'] * np.outer(np.cos(u), np.sin(v)) + obstacle['x']
            y = obstacle['radius'] * np.outer(np.sin(u), np.sin(v)) + obstacle['y']
            z = obstacle['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle['z']
            ax.plot_surface(x, y, z, color='black', alpha=0.5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title('All Agent Trajectories')
        plt.show()

    def visualize_metrics(self):
        """
        可视化 时间、燃油量、危险系数 变化
        """
        
        epochs = range(len(self.trace_time))
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        
        # Time plot
        axs[0].plot(epochs, self.trace_time, label='Time', color='blue')
        axs[0].set_title('Time per Epoch')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Time')
        axs[0].legend()
        
        # fuel plot
        axs[1].plot(epochs, self.trace_fuelcost, label='Fuel', color='Yellow')
        axs[1].set_title('Time per Epoch')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('fuel')
        axs[1].legend()

        # Danger plot
        axs[2].plot(epochs, self.trace_danger, label='Danger', color='red')
        axs[2].set_title('Danger per Epoch')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Danger')
        axs[2].legend()
        
        plt.tight_layout()
        plt.savefig(self.image_save_path + '-metrics' + '.png', format='png', dpi=300)
        plt.show()
