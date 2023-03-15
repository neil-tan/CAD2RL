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
class DQN(nn.Module):
    def __init__(self, num_states, num_actions):
        super(DQN, self).__init__()
        self.num_states = num_states
        self.module = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=num_states, stride=1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16, num_actions),
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
      x = x.reshape(-1, 1, self.num_states)
      out = self.module(x)
      return out
# %%
class DeepQLearning:
  def __init__(self, env_maker, memory_size=2000, batch_size=64):
    self.memory_size = memory_size
    self.batch_size = batch_size
    self.memory_bank = MemoryBank(capacity=memory_size)
    self.action_network = DQN(num_states=4, num_actions=2)
    self.target_network = DQN(num_states=4, num_actions=2)
    self.sync_target_network()
    self.animator = AgentAnimator(env_maker, num_agents=24, q_function=self.q_function)

    self.sync_target_network()
    
    with torch.no_grad():
      self.animator.fill(self.memory_bank, epsilon=0.85)
    # self.data_loader = DataLoader(self.memory_bank, batch_size=batch_size, shuffle=True)


  # TODO: RUN THIS
  def train(self):
    gamma = 0.999
    lr = 0.001
    copy_iter = 5
    n_episode = 1000

    loss_func = torch.nn.MSELoss()
    # loss_func = torch.nn.SmoothL1Loss()

    # optimizer = torch.optim.SGD(self.action_network.parameters(), lr=lr)
    # optimizer = torch.optim.RMSprop(self.action_network.parameters())
    optimizer =  torch.optim.Adam(self.action_network.parameters(), lr=lr)

    past_episode_max_len = False
    max_len = 0
    copy_iter_average_len = 0
    print("n_episode: ", n_episode)
    for i in range(n_episode):
      with torch.no_grad():
        b_state, b_action, b_reward, b_new_state = self.memory_bank.sample(self.batch_size)

        qs_target = self.target_network(b_new_state)
        y = b_reward + gamma * qs_target.max(1)[0]
        y = y.detach()

      qs_action = self.action_network(b_state)
      q = torch.gather(qs_action, dim=1, index=b_action.long().unsqueeze(1)).squeeze(1)
        
      loss = loss_func(q, y)

      loss.backward()
      for param in self.action_network.parameters():
        param.grad.data.clamp_(-1, 1)
      optimizer.step()
      optimizer.zero_grad()

      batch_average_trace_lens = self.animator.fill(self.memory_bank, num_runs=self.batch_size, epsilon=max(0.5*(1 - i/50), 0.005))

      copy_iter_average_len += batch_average_trace_lens
      
      if batch_average_trace_lens > max_len:
        max_len = batch_average_trace_lens
        self.save_best_param()
        past_episode_max_len = True

      if i % copy_iter == 0:
        self.sync_target_network()
        if past_episode_max_len:
          print(colored("rolling reward: " + str(copy_iter_average_len/copy_iter) + " sync loss: " + str(loss.item()) + " max: " + str(max_len), 'red'))
          past_episode_max_len = False
        else:
          print("rolling reward: ", copy_iter_average_len/copy_iter, "sync loss: ", loss.item())
        copy_iter_average_len = 0
      
    self.load_best_param()
    print("max len: ", max_len)
  
  def sync_target_network(self):
    with torch.no_grad():
      self.target_network.load_state_dict(self.action_network.state_dict())

  def save_best_param(self):
    with torch.no_grad():
      self.beset_param = copy.deepcopy(self.action_network.state_dict())

  def load_best_param(self):
    with torch.no_grad():
      self.action_network.load_state_dict(self.beset_param)

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
