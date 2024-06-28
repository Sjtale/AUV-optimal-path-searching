import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from gym.spaces import Discrete, Box
from gym import Env
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
import heapq

# 检查是否有可用的GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
class ReplayBuffer_random:
    """
    随机 BUFFER
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)



class ReplayBuffer_prioritise:
    """
    Prioritise Buffer
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)  # Optional for prioritized experience replay

    def push(self, state, action, reward, next_state, done):
        priority = reward
        self.buffer.append((state, action, reward, next_state, done))
        if reward < 0:
            priority = reward / -200
        self.priorities.append(priority)

    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]

        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)
    

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        features = self.feature(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        q_value = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_value
    
    
def preprocess_state(state):
    return torch.FloatTensor(state).unsqueeze(0).to('cpu')

class DQNAgent:
    def __init__(self, use_prioritise_buffer, state_dim, action_dim, buffer_capacity=400, batch_size=32, gamma=1, lr=0.001):
        self.action_dim = action_dim
        if use_prioritise_buffer:
            self.memory = ReplayBuffer_prioritise(buffer_capacity)
        else:
            self.memory = ReplayBuffer_random(buffer_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = 1
        self.epsilon_decay = 0.996
        self.epsilon_min = 0.01
        self.epsilon_decay_win = 1
        self.model = DuelingDQN(state_dim, action_dim).to(device)
        self.target_model = DuelingDQN(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.update_target_network()
    
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = preprocess_state(state)
        q_values = self.model(state)
        return torch.argmax(q_values, dim=1).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        current_q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        target_q_values = target_q_values.unsqueeze(1)

        # loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def save_model(self, model_filename,target_model_filename):
        torch.save(self.model.state_dict(), model_filename)
        torch.save(self.target_model.state_dict(),target_model_filename)
        
 
