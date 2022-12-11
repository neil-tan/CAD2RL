# %%
import jax
import jax.numpy as jnp
import numpy as np

from collections import namedtuple
import copy
import asyncio
import concurrent.futures
from termcolor import colored

import gym
from common.jupyter_animation import animate, animation_table
from common.memory_bank import MemoryBank
from common.agents import AgentAnimator, T_Element
from typing import Iterable, Union, Callable, Tuple, List, Dict, Any
import time
import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from wrapt import synchronized

# %%
def animate_policy(policy:callable):
  env = gym.make('CartPole-v1', render_mode="rgb_array")
  state, info = env.reset()
  image_seq = []

  done = False
  while not done:
    state, reward, done, truncated, info = env.step(policy(state))
    image_seq.append(env.render())

  return animate(image_seq)

# %%
animate_policy(lambda state: np.random.randint(0,1))

# %%
class MLP(nn.Module):
    def __init__(self, num_states, hidden_dim, num_actions, dropout=0.5):
        super(MLP, self).__init__()
        self.num_states = num_states
        self.module = nn.Sequential(
            nn.Linear(num_states, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    

    def forward(self, x):
      x = x.reshape(-1, 1, self.num_states)
      out = self.module(x)
      return out
# %%
class VPG:
  def __init__(self, env_maker, memory_size=1000, batch_size=1):
    self.memory_size = memory_size
    self.batch_size = batch_size
    self.memory_bank = MemoryBank(capacity=memory_size)
    self.policy_network = MLP(num_states=4, hidden_dim=128, num_actions=2)
    self.env_maker = env_maker

  def train(self):
    lr = 0.01
    print_iter = 1000
    n_episode = 15000
    max_reward = 0

    # loss_func = torch.nn.SmoothL1Loss()

    optimizer =  torch.optim.Adam(self.policy_network.parameters(), lr=lr)

    accumlated_reward = 0
    past_episode_max_reward = False
    print("n_episode: ", n_episode)
    for i in range(1, n_episode+1):      

      animator = AgentAnimator(self.env_maker, num_agents=1, q_function=self.select_action)
      animator.fill(self.memory_bank, num_runs=1, epsilon=0)

      b_state, b_action, b_reward, b_new_state, discounted_reward = self.memory_bank.sample(len(self.memory_bank))
      discounted_reward = discounted_reward.detach()

      at_st = self.policy_network(b_state)
      loss = torch.log(at_st) * discounted_reward.unsqueeze(1)
      loss = - loss.sum()

      loss.backward()

      for param in self.policy_network.parameters():
        param.grad.data.clamp_(-1, 1)

      optimizer.step()
      optimizer.zero_grad()

      self.memory_bank.clear_memory()


      reward = b_reward.sum().item()
      accumlated_reward += reward
      
      if reward > max_reward:
        max_reward = reward
        self.save_best_param()
        past_episode_max_reward = True
        
      if i % print_iter == 0:
        if past_episode_max_reward:
          print(colored("rolling reward: " + str(accumlated_reward/print_iter) + " max: " + str(max_reward), 'red'))
          past_episode_max_reward = False
        else:
          print("rolling reward: ", accumlated_reward/print_iter)
        accumlated_reward = 0

    self.load_best_param()
    print("max reward: ", max_reward)

  
  def calculate_returns(self, rewards, discount_factor, normalize = True):
    returns = []
    R = 0
    
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
        
    returns = torch.tensor(returns)
    
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
        
    return returns


  def save_best_param(self):
    with torch.no_grad():
      self.best_param = copy.deepcopy(self.policy_network.state_dict())

  def load_best_param(self):
    with torch.no_grad():
      self.policy_network.load_state_dict(self.best_param)

  @synchronized
  def select_action(self, state):
    input = torch.tensor(state)
    output = self.policy_network(input)
    action_prob = F.softmax(output, dim=-1)
    output = output.detach().numpy()
    action = int(np.argmax(output))
    return action


# %%
test_obj = VPG(env_maker=lambda: gym.make('CartPole-v1', render_mode="rgb_array"))
test_obj.train()

print("Training finished")
# %%
animate_policy(lambda state: test_obj.select_action(state))
# %%
