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
from common.rl_common import sample_trajectory, discounted_rewards
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

        self.apply(self.init_weights)

    def init_weights(self, m):
      if type(m) == nn.Linear:
          torch.nn.init.xavier_normal_(m.weight)
          m.bias.data.fill_(0.01)

    def forward(self, x):
      x = x.reshape(-1, self.num_states)
      out = self.module(x)
      return out
# %%
class VPG:
  def __init__(self, env_maker, batch_size=1):
    self.policy_network = MLP(num_states=4, hidden_dim=32, num_actions=2, dropout=0.7)
    self.env = env_maker()
  
  def train(self):
    lr = 0.01
    n_episode = 2500
    print_iter = 100
    discount_factor = 0.99
    gradient_clip = 1

    self.policy_network.train()
    optimizer =  torch.optim.Adam(self.policy_network.parameters(), lr=lr)

    accumlated_reward = 0
    max_reward = 0
    past_episode_max_reward = False


    print("n_episode: ", n_episode)
    for i in range(1, n_episode+1):    
      _, _, log_prob_actions, rewards = sample_trajectory(self.env, self.policy_network, max_steps=1000)

      # return a list of discounted rewards [steps]
      returns = discounted_rewards(rewards, discount_factor=discount_factor)

      # Update Policy
      returns = returns.detach() # [steps]
      loss = -log_prob_actions * returns
      loss = loss.sum()

      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), gradient_clip)
      optimizer.step()
      optimizer.zero_grad()

      trace_reward = sum(rewards)
      accumlated_reward += trace_reward
      
      if trace_reward > max_reward:
        max_reward = trace_reward
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


  def save_best_param(self):
    with torch.no_grad():
      self.best_param = copy.deepcopy(self.policy_network.state_dict())

  def load_best_param(self):
    with torch.no_grad():
      self.policy_network.load_state_dict(self.best_param)

  def select_action(self, state):
    with torch.no_grad():
      input = torch.tensor(state)
      output = self.policy_network(input).detach().numpy()
      action = int(np.argmax(output))
      return action

# %%
test_obj = VPG(env_maker=lambda: gym.make('CartPole-v1', render_mode="rgb_array"))
test_obj.train()

print("Training finished")
# %%
animate_policy(lambda state: test_obj.select_action(state))
# %%
