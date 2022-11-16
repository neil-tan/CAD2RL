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
from common.agents import AgentAnimator, Q_Element
from typing import Iterable, Union, Callable, Tuple, List, Dict, Any
import time
import random

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
class DQN(nn.Module):
    def __init__(self, num_states, num_actions):
        super(DQN, self).__init__()
        self.num_states = num_states
        self.module = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=num_states, stride=1),
            nn.Conv1d(32, 128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, num_actions),
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
      x = x.reshape(-1, 1, self.num_states)
      out = self.module(x)
      return out
# %%
class DeepQLearning:
  def __init__(self, env_maker, memory_size=10000, batch_size=128):
    self.memory_size = memory_size
    self.batch_size = batch_size
    self.memory_bank = MemoryBank(capacity=memory_size)
    self.action_network = DQN(num_states=4, num_actions=2)
    self.target_network = DQN(num_states=4, num_actions=2)
    self.sync_target_network()

    self.loop = asyncio.get_event_loop()
    self.env = env_maker()

    with torch.no_grad():
      animator = AgentAnimator(env_maker, num_agents=24, q_function=self.q_function)
      animator.fill(self.memory_bank, epsilon=1)
    # self.data_loader = DataLoader(self.memory_bank, batch_size=batch_size, shuffle=True)

  # TODO: check here
  def select_action(self, state, epsilon=0.3):
    with torch.no_grad():
      if random.random() > epsilon:
        with torch.no_grad():
          result = self.q_function(state)
          return result
      else:
        return self.env.action_space.sample()


  def train(self):
    gamma = 0.999
    # gamma = 1
    lr = 0.01
    copy_iter = 10
    n_episode = int(self.memory_size / self.batch_size * 50)
    max_reward = 0

    loss_func = torch.nn.MSELoss()
    # loss_func = torch.nn.SmoothL1Loss()

    # optimizer = torch.optim.SGD(self.action_network.parameters(), lr=lr)
    # optimizer = torch.optim.RMSprop(self.action_network.parameters())
    optimizer =  torch.optim.Adam(self.action_network.parameters(), lr=lr)

    past_episode_max_reward = False
    print("n_episode: ", n_episode)
    for i in range(1, n_episode+1):
      state, info = self.env.reset()
      
      done, truncated, info = False, False, None
      total_reward = 0
      copy_iter_average_reward = 0
      while not done and not truncated:
        with torch.no_grad():
          action = self.select_action(state, epsilon=0.3*(1 - i/n_episode))
          new_state, reward, done, truncated, info = self.env.step(action)
          self.memory_bank.add(Q_Element(state, action, reward, new_state))
          state = new_state

          b_state, b_action, b_reward, b_new_state = self.memory_bank.sample(self.batch_size)
         
          qs_target = self.target_network(b_new_state)
          y = b_reward + gamma * qs_target.max(1)[0]
          y = y.detach()

        qs_action = self.action_network(b_state)
        # q = torch.gather(qs_action, dim=1, index=b_action.unsqueeze(1)).squeeze(1)
        q = qs_action.gather(1, b_action.long().unsqueeze(-1)).squeeze(-1)
        # y = torch.ones_like(y)
          
        loss = loss_func(q, y)

        loss.backward()
        for param in self.action_network.parameters():
          param.grad.data.clamp_(-1, 1)
        optimizer.step()
        optimizer.zero_grad()
        

        total_reward += reward
        copy_iter_average_reward += reward
      
      if total_reward > max_reward:
        max_reward = total_reward
        # self.sync_target_network()
        past_episode_max_reward = True
        
      if i % copy_iter == 0:
        self.sync_target_network()
        if past_episode_max_reward:
          print(colored("rolling reward: " + str(copy_iter_average_reward) + " sync loss: " + str(loss.item()) + " max: " + str(max_reward), 'red'))
          past_episode_max_reward = False
        else:
          print("rolling reward: ", copy_iter_average_reward, "sync loss: ", loss.item())
        copy_iter_average_reward = 0
  
    print("max reward: ", max_reward)

  
  def sync_target_network(self):
    with torch.no_grad():
      self.target_network.load_state_dict(self.action_network.state_dict())

  @synchronized
  def q_function(self, state):
    with torch.no_grad():
      input = torch.tensor(state)
      output = self.action_network(input)
      output = output.detach().numpy()
      action = int(np.argmax(output))
      return action


# %%
test_obj = DeepQLearning(env_maker=lambda: gym.make('CartPole-v1', render_mode="rgb_array"))
test_obj.train()

print("Training finished")
# %%
animate_policy(lambda state: test_obj.q_function(state))

# %%
