import random
import gymnasium as gym
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple
from torch.distributions import Categorical

#hyperparameter
BUFFER_SIZE = int(5e4)
EPISODE = int(1e4)
BATCH_SIZE = 128
TAU = 1e-3

LEARNING_RATE = 1e-4
GAMMA = 0.99

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayBuffer:
    def __init__(self, max_length):
        self.buffer = collections.deque(maxlen=max_length)
        
    def put_data(self,*args):
        self.buffer.append(Transition(*args))
        
    def sample_minibatch(self):
        return random.sample(self.buffer, BATCH_SIZE)
        
    def __len__(self):
        return len(self.buffer)
    
class Qnetwork(nn.Module):
    def __init__(self, obs_space_dims:int, action_space_dims:int):
        super(Qnetwork,self).__init__()
        
        hidden_layer1 = 128
        hidden_layer2 = 128
        
        self.net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_layer1),
            nn.ReLU(),
            nn.Linear(hidden_layer1, hidden_layer2),
            nn.ReLU(),
            nn.Linear(hidden_layer2, action_space_dims)
        )
    
    def forward(self, state:np.ndarray):
        q_value = self.net(state)
        return q_value
        
class DQN:
    def __init__(self, obs_space_dims:int, action_space_dims:int):

        self.action_space_dims = action_space_dims
        
        self.qNet = Qnetwork(obs_space_dims, action_space_dims)
        self.qtargetNet = Qnetwork(obs_space_dims, action_space_dims)
        self.qtargetNet.load_state_dict(self.qNet.state_dict()) # copy
        self.optimizer = optim.AdamW(self.qNet.parameters(), lr=LEARNING_RATE)
        
        self.buffer = ReplayBuffer(BUFFER_SIZE)

    def sample_action(self, state:torch.Tensor, threshold:float):
        coin = random.random()
        if coin > threshold:
            with torch.no_grad():
                return self.qNet(state).max(1)[1].view(1,1)
        else:
            sample = random.sample([0,1], 1)
            return torch.tensor([sample],dtype=torch.long)
            
    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return
        transitions = self.buffer.sample_minibatch() # transition은 tuple로 이루어진 list
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                 batch.next_state)), dtype=torch.bool)
        non_final_next_state = torch.cat([s for s in batch.next_state
                                             if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        q_values = self.qNet(state_batch).gather(1, action_batch)
        
        next_q_values = torch.zeros(BATCH_SIZE)
        with torch.no_grad():
            next_q_values[non_final_mask] = self.qtargetNet(non_final_next_state).max(1)[0]
            
        target = reward_batch + GAMMA * next_q_values
        loss = F.smooth_l1_loss(target.unsqueeze(1), q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 목표 네트워크의 가중치를 소프트 업데이트
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.qtargetNet.state_dict()
        policy_net_state_dict = self.qNet.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.qtargetNet.load_state_dict(target_net_state_dict)
        
def main():
    
    env = gym.make('CartPole-v1', render_mode = 'human') # 학습하는 것을 보고 싶으시다면
    #env = gym.make('CartPole-v1')
    observation_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.n
    
    for seed in [2,3,5]:
        
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        agent = DQN(observation_space_dims, action_space_dims)
        
        # 출력변수
        print_interval = 50
        score = 0.0
        
        for n_episode in range(EPISODE):
            state, _ = env.reset(seed=seed)
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0) # size [1,4]
            epsilon = max(0.01, 0.08 - 0.01*(n_episode/200)) #Linear annealing from 8% to 1% (n_episod = 1400 -> 1%)
            done = False
            
            while not done:
                action = agent.sample_action(state, epsilon) # size[1,1]
                observation, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                reward = torch.tensor([reward]) # size [1]
                
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0) # size [1,4]
                
                agent.buffer.put_data(state,action,reward,next_state)
                agent.update()
                
                state = next_state
                score += reward.item()
                
                if done:
                    break
                
            
            if n_episode % print_interval == 0 and n_episode != 0:
                print("seed : %d, episode = %d, avg_score : %.2f"
                      %(seed, n_episode, score/print_interval))
                score = 0

if __name__ == '__main__':
    main()
    
    
# reference
# 1. https://tutorials.pytorch.kr/intermediate/reinforcement_q_learning.html
# 2. minimalRL dqn.py