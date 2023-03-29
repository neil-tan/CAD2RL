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
from common.rl_common import batch_trajectory, discounted_rewards
from typing import Iterable, Union, Callable, Tuple, List, Dict, Any
import time
import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

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
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, in_dim, hidden_dim, out_dim):
		"""
			Initialize the network and set up the layers.

			Parameters:
				in_dim - input dimensions as an int
        hidden_dim - hidden dimensions as an int
				out_dim - output dimensions as an int

			Return:
				None
		"""
		super(MLP, self).__init__()

		self.layer1 = nn.Linear(in_dim, hidden_dim)
		self.layer2 = nn.Linear(hidden_dim, hidden_dim)
		self.layer3 = nn.Linear(hidden_dim, out_dim)

	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		return output

# %%
class PPO:
  def __init__(self, env_maker, batch_size=64, lr=0.001, gamma=0.99, clip=0.2, epochs=1000, n_ppo_updates=3, max_steps=1000):
    # Initialize hyperparameters
    self.batch_size = batch_size
    self.lr = lr
    self.gamma = gamma
    self.clip = clip
    self.epochs = epochs
    self.n_ppo_updates = n_ppo_updates
    self.max_steps = max_steps
  
    # Initialize environment
    self.env = env_maker()

    # Initialize actor and critic models
    self.actor_model = MLP(in_dim=4, hidden_dim=64, out_dim=2)
    self.critic_model = MLP(in_dim=4, hidden_dim=64, out_dim=1)

    # Initialize optimizers for actor and critic
    self.actor_optim = Adam(self.actor_model.parameters(), lr=self.lr)
    self.critic_optim = Adam(self.critic_model.parameters(), lr=self.lr)
          
    # Initialize memory bank
    self.memory_bank = MemoryBank()

  def V(self, state, critic:Callable=None):
    critic = self.critic_model if critic is None else critic
    state = torch.cat(state) if isinstance(state, list) else state
    return critic(state) # [batch, 1]
  
  def log_action_prob(self, state, action, policy:Callable=None):
    policy = self.actor_model if policy is None else policy
    state = torch.cat(state) if isinstance(state, list) else state
    action = torch.cat(action) if isinstance(action, list) else action

    action_pred = policy(state) # num_actions [batch, action dims] -> [1, 2]
    action_prob = F.softmax(action_pred, dim=-1) # num_actions [batch, 2]
    dist = torch.distributions.Categorical(action_prob) # [batch, 2]
    log_prob_action = dist.log_prob(action) # log(action_prob[action]) -> [batch] # probability of action taken

    return log_prob_action

  def update_PPO(self, observations, actions, log_action_probs, rewards, advantages, clip, epochs):
    for _ in range(epochs):
      V = self.V(observations)
      new_action_log_probs = self.log_action_prob(observations, actions)
      ratio = torch.exp(new_action_log_probs - log_action_probs) # [batch]

      # surrogate losses
      surr1 = ratio * advantages
      surr2 = torch.clamp(ratio, 1-clip, 1+clip) * advantages

      # actor loss
      actor_loss = -torch.min(surr1, surr2).mean()

      # critic loss
      critic_loss = nn.MSELoss()(V.squeeze(), rewards)

      # Updating the actor and critic
      self.actor_optim.zero_grad()
      actor_loss.backward(retain_graph=True)
      self.actor_optim.step()

      self.critic_optim.zero_grad()
      critic_loss.backward()
      self.critic_optim.step()

  def train(self):
    print_iter = 100

    self.actor_model.train()
    self.critic_model.train()

    accumlated_reward = 0
    max_reward = 0
    past_episode_max_reward = False

    print("n_episode: ", self.epochs)
    for i in range(1, self.epochs+1):    
      # compute the trajectory usin the current policy
      batch_obs, batch_acts, batch_log_probs, rewards = batch_trajectory(self.env, self.actor_model, batch_size=self.batch_size, max_steps=self.max_steps)
      batch_reward_otgs = discounted_rewards(rewards, discount_factor=self.gamma)

      V = self.V(batch_obs)
      A_k = rewards - V.detach()
      A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) # NORMALIZE

      # return a list of discounted rewards [steps]
      

      # PPO Update
      self.update_PPO(batch_obs, batch_acts, batch_log_probs, batch_reward_otgs, A_k, self.clip, epochs=5)

      trace_reward = sum(rewards) / self.batch_size
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
      self.best_param = copy.deepcopy(self.actor_model.state_dict())

  def load_best_param(self):
    with torch.no_grad():
      self.actor_model.load_state_dict(self.best_param)

  def select_action(self, state):
    with torch.no_grad():
      input = torch.tensor(state)
      output = self.actor_model(input).detach().numpy()
      action = int(np.argmax(output))
      return action

# %%
test_obj = PPO(env_maker=lambda: gym.make('CartPole-v1', render_mode="rgb_array"))
test_obj.train()

print("Training finished")
# %%
animate_policy(lambda state: test_obj.select_action(state))
# %%
