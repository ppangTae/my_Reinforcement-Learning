import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import numpy as np
import gymnasium as gym

# hyperparameters
LR_ACTOR = 1e-3
LR_CRITIC = 5e-3
BATCH_SIZE = 64
BUFFER_SIZE = int(1e6)
GAMMA = 0.99
TAU = 5e-3
START_TIMESTEPS = 25e2
MAX_TIMESTEPS = 1e6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, reward, next_state):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
        # 0 ~ self.size에서 batch_size만큼 샘플링해주는 함수(dtype=int)
		ind = np.random.randint(0, self.size, size=batch_size) 

		return (
			torch.FloatTensor(self.state[ind]).to(self.device), # FloatTensor의 기본형은 torch.float32
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device)
        )

class ActorNet(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorNet, self).__init__()   
        
        hidden_layer1 = 256
        hidden_layer2 = 256
        
        self.fc1 = nn.Linear(obs_dim, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.fc3 = nn.Linear(hidden_layer2 ,action_dim)
        
    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = 2 * torch.tanh(self.fc3(a))
        return a
        
class CriticNet(nn.Module):
    
    def __init__(self, obs_dim, action_dim):
        super(CriticNet, self).__init__()
        
        hidden_layer1 = 256
        hidden_layer2 = 256
        
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.fc3 = nn.Linear(hidden_layer2 ,1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        q = F.relu(self.fc1(cat))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q

class DDPG:
    
    def __init__(self, obs_dim, action_dim):
        self.buffer = ReplayBuffer(obs_dim, action_dim, BUFFER_SIZE)
        
        self.criticNet = CriticNet(obs_dim, action_dim).to(device)
        self.actorNet = ActorNet(obs_dim, action_dim).to(device)
        self.criticTargetNet = CriticNet(obs_dim, action_dim).to(device)
        self.actorTargetNet = ActorNet(obs_dim, action_dim).to(device)
        self.criticTargetNet.load_state_dict(self.criticNet.state_dict())
        self.actorTargetNet.load_state_dict(self.actorNet.state_dict())
        
        self.critic_optimizer = optim.Adam(self.criticNet.parameters(), lr=LR_CRITIC)
        self.actor_optimizer = optim.Adam(self.actorNet.parameters(), lr=LR_ACTOR)
        
    def sample_action(self, state:np.ndarray):
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            action = self.actorNet(state)
        return action.cpu().numpy()
        
    def soft_target_net_update(self):
        actor_net_state_dict = self.actorNet.state_dict()
        critic_net_state_dict = self.criticNet.state_dict()
        actor_target_net_state_dict = self.actorTargetNet.state_dict()
        critic_target_net_state_dict = self.criticTargetNet.state_dict()
        
        for a_key, c_key in zip(actor_net_state_dict, critic_net_state_dict):
            actor_target_net_state_dict[a_key] = actor_net_state_dict[a_key]*TAU + actor_target_net_state_dict[a_key]*(1-TAU)
            critic_target_net_state_dict[c_key] = critic_net_state_dict[c_key]*TAU + critic_target_net_state_dict[c_key]*(1-TAU)
            
        self.actorTargetNet.load_state_dict(actor_target_net_state_dict)
        self.criticTargetNet.load_state_dict(critic_target_net_state_dict)
        
    def update(self):
        
        state, action, reward, next_state = self.buffer.sample(BATCH_SIZE)
        
        q_values = self.criticNet(state, action) # size [BATCH_SIZE,1]
        
        with torch.no_grad():
            next_q_values= self.criticTargetNet(next_state, self.actorTargetNet(next_state))
            target = reward + GAMMA * next_q_values
            
        self.critic_optimizer.zero_grad()
        critic_loss = F.smooth_l1_loss(q_values, target)
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        actor_loss = -self.criticNet(state, self.actorNet(state)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_target_net_update()
        
# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(actor, seed, eval_episodes=10):
    eval_env = gym.make('Pendulum-v1')
    
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = eval_env.reset(seed=seed)
        done = False
        
        while not done:
            
            action = actor(torch.from_numpy(state).float().to(device))
            state, reward, terminated, truncated , _ = eval_env.step(action)
            done = terminated, truncated
            avg_reward += reward
            
        avg_reward /= eval_episodes
        
        print('---------------------------------------')
        print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
        print("---------------------------------------")
        
        return avg_reward
        
        
def main():
    env = gym.make('Pendulum-v1')
    #env = gym.make('Pendulum-v1', g=9.81, render_mode = 'human') # 학습하는 것을 보고싶다면
    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    expl_noise = 0.1
    
    # 출력 설정
    eval_freq = 1000
    
    # seed 설정
    seed = 117 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    state, _ =env.reset(seed=seed)
    episode_timesteps = 0
    episode_num = 0
        
    agent = DDPG(obs_space_dims, action_space_dims)
    score = 0.0
        
    for t in range(int(MAX_TIMESTEPS)):
        
        episode_timesteps += 1
                
        if t < START_TIMESTEPS: # 학습 전 충분한 데이터를 모으기 위해서.
            action = env.action_space.sample() # action은 numpy array
        else:
            action = (
                agent.sample_action(state)
                + np.random.normal(0, max_action * expl_noise, size=action_space_dims)
            ).clip(-max_action, max_action)
            
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        next_state = observation if not terminated else None
        
        agent.buffer.add(state, action, reward, next_state)
        agent.update()
        
        score += reward
        state = next_state
        
        if done:
            state, _ = env.reset(seed=seed)
            episode_num += 1
            
        if (t + 1) % eval_freq == 0:
            print(f"time_step : {t+1} --> avg_score : {score/eval_freq}")
            score = 0.0
            

if __name__ == '__main__':
    main()