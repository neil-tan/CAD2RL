# %%
import jax
import jax.numpy as jnp
import numpy as np

from collections import namedtuple
import copy
import asyncio
import concurrent.futures

import gym
from common.jupyter_animation import animate, animation_table
from common.memory_bank import MemoryBank
from common.agents import Agent, AgentAnimator
from typing import Iterable, Union, Callable, Tuple, List, Dict, Any
import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
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
class DeepQLearning:
  def __init__(self, env_maker, memory_size=10000, batch_size=128):
    self.memory_size = memory_size
    self.batch_size = batch_size
    self.memory_bank = MemoryBank(capacity=memory_size)
    self.action_network = self.create_network(num_states=4, num_actions=2)
    self.target_network = self.create_network(num_states=4, num_actions=2)
    self.sync_target_network()
    self.animator = AgentAnimator(env_maker, num_agents=24, q_function=self.q_function)

    self.action_network.float()
    self.target_network.float()

    self.loop = asyncio.get_event_loop()

    self.animator.fill(self.memory_bank)
    # self.data_loader = DataLoader(self.memory_bank, batch_size=batch_size, shuffle=True)

  # input: self.Q_Element = namedtuple('Q_Entry', ['state', 'action', 'reward', 'new_state'])
  # def forward(self, state):
  #   target_q = self.target_network(state)
  #   action_q = self.action_network(state)
  #   loss_func = torch.nn.MSELoss()
  #   loss = loss_func(action_q, target_q)


  # TODO: RUN THIS
  def train(self):
    gamma = 0.999
    lr = 0.1
    copy_epoch = 10

    # loss_func = torch.nn.MSELoss()
    loss_func = torch.nn.SmoothL1Loss()
    self.target_network.train(False)
    self.action_network.train(True)

    optimizer = torch.optim.SGD(self.action_network.parameters(), lr=lr)

    average_epoch_loss = 0

    for i in range(1000):
      state, action, reward, new_state = self.memory_bank.sample(self.batch_size)

      reward = reward.float()
      qs_target = self.target_network(new_state)
      y = reward + gamma * torch.max(qs_target, dim=1).values
      qs_action = self.action_network(state)
      q = torch.gather(qs_action, dim=1, index=action.unsqueeze(1)).squeeze(1)
        
      loss = loss_func(y, q)
      average_epoch_loss += loss.item()

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      self.animator.fill(self.memory_bank, num_runs=self.batch_size)
      
      if i % copy_epoch == 0:
        self.sync_target_network()
        print(average_epoch_loss/copy_epoch)
      
      average_epoch_loss = 0
  
  def sync_target_network(self):
    self.target_network.load_state_dict(self.action_network.state_dict())

  @synchronized
  def q_function(self, state):
    input = torch.tensor(state)
    output = self.action_network(input)
    output = output.detach().numpy()
    action = int(np.argmax(output))
    return action

  def create_network(self, num_states, num_actions):
    return nn.Sequential(
      nn.Linear(num_states, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, num_actions),
      nn.ReLU()
    )


# %%
test_obj = DeepQLearning(env_maker=lambda: gym.make('CartPole-v1', render_mode="rgb_array"))
test_obj.train()

print("Training finished")
# %%
animate_policy(lambda state: test_obj.q_function(state))

# %%
