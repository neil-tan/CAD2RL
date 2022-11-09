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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# %%
env = gym.make('CartPole-v1', render_mode="rgb_array")
env.reset()
image_seq = []

# %%
done = False
while not done:
  observation, reward, done, truncated, info = env.step(env.action_space.sample())
  image_seq.append(env.render())

animate(image_seq)

# %%
env.reset()[0]

# %%

class DeepQLearning:
  def __init__(self, env_maker, memory_size=200000):
    self.memory_bank = MemoryBank(memory_size=memory_size)
    self.action_network = self.create_network(num_states=4, num_actions=2)
    self.target_network = self.create_network(num_states=4, num_actions=2)
    self.animator = AgentAnimator(env_maker, num_agents=16, q_function=self.q_function)

    self.animator.fill(self.memory_bank)
    self.data_loader = DataLoader(self.memory_bank, batch_size=32, shuffle=True)
  
  def train(self):
    for batch in self.data_loader:
      print(batch)
      break


  def q_function(self, state):
    # need to improve the efficiency of this
    loop = asyncio.get_event_loop()
    future = loop.call_soon_threadsafe(self.action_network, state)
    return future.result()

  def create_network(self, num_states, num_actions):
    return nn.Sequential(
      nn.Linear(num_states, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, num_actions),
      nn.ReLU()
    )

# 

# %%
def discretize_state(space_obj, num_bins:Iterable[int], return_indexi=False, max_value=16) -> Callable:
  bins = [np.linspace(max(low, -max_value), min(high, max_value), num=num_bin) for low, high, num_bin in zip(space_obj.low, space_obj.high, num_bins)]

  def discretize_state_fn(input_state:np.ndarray) -> np.ndarray:
    discretized_state = np.zeros(input_state.shape)
    discretized_indexi = np.zeros(input_state.shape, dtype=np.int32)
    for i, bin_space in enumerate(bins):
      discrete_index = np.digitize(input_state[i], bin_space)-1
      discretized_state[i] = bins[i][discrete_index]
      discretized_indexi[i] = discrete_index
    
    ret = discretized_state if not return_indexi else discretized_indexi
    ret = tuple(ret)
    return ret

  return discretize_state_fn

# discret_func = discretize_state(env.observation_space, num_bins=(10, 10, 10, 10))
# discret_func(np.array([-4.8, -100, 1, -10]))
# discret_func(np.array([5, 2000, 0.1, 0]))

# %%
# Hyperparameters
alpha = 0.1
gamma = 0.95
epsilon_curiosity = 0.15
max_epoch = 55000

discret_func = discretize_state(env.observation_space, num_bins=(50, 50, 80, 80), max_value=12, return_indexi=True)
q_table = np.zeros((50, 50, 80, 80, 2))
all_steps_taken = []
all_rewards = []


for i in range(1, max_epoch):
  state = discret_func(env.reset()[0])
  done = False
  steps_taken = 0
  total_reward = 0

  # trace
  while not done:
    if np.random.uniform(0, 1) < epsilon_curiosity:
      action = env.action_space.sample()
    else:
      action = int(np.argmax(q_table[state]))

    raw_state, reward, done, truncated, info = env.step(action)
    next_state = discret_func(raw_state)

    # if state[0] >= 30:
    #   print("dugging")

    q_value = q_table[state + (action,)]
    next_max = np.max(q_table[next_state])

    new_q_value = (1 - alpha) * q_value + alpha * (reward + gamma * next_max)
    q_table[state + (action,)] = new_q_value

    state = next_state
    steps_taken += 1
    total_reward += reward
  
  all_steps_taken.append(steps_taken)
  all_rewards.append(total_reward)
  
  if i % 1000 == 0:
    print("[{n}] Average (10 epochs), steps: {steps}, reward: {reward}".format(n=i, steps=sum(all_steps_taken[-10:]) / 10, reward=sum(all_rewards[-10:]) / 10))


print("Training finished")


# %%
print("total elements: ", q_table.size)
print("q vlaue >= 0.5: ", np.count_nonzero(q_table >= 0.5))
print("q vlaue < 0.5: ", np.count_nonzero(q_table < 0.5))

# %%
# Test Helper
def sample_from_q_table(q_table, state, epsilon_curiosity=0.1):
  if np.random.uniform(0, 1) < epsilon_curiosity:
    action = env.action_space.sample()
  else:
    action = int(np.argmax(q_table[state]))
  return action

# Animate the agent
image_seq = []
done = False
state = env.reset()[0]
while not done:
  action = sample_from_q_table(q_table, discret_func(state), epsilon_curiosity=0)
  state, reward, done, truncated, info = env.step(action)
  image_seq.append(env.render())

animate(image_seq)




# %%
