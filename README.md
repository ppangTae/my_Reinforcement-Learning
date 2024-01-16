1. DQN
   I refer to the codes below
   1. https://tutorials.pytorch.kr/intermediate/reinforcement_q_learning.html
   2. [minimalRL dqn code](https://github.com/seungeunrho/minimalRL/blob/master/dqn.py)
2. DDPG
   I refer to the code below
   https://github.com/sfujim/TD3/blob/master/utils.py

The replay buffer for DQN and DDPG is different.
DQN replaybuffer is so slow. because it has many copy.
so, if you want to implement replaybuffer, please use ddpg's replaybuffer
  
