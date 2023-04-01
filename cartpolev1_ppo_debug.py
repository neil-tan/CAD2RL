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
import torch.distributions as distributions
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

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout = 0.5):
        super().__init__()

        self.fc_1 = nn.Linear(in_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc_2(x)
        return x

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0)

# %%
def update_policy(networks, states, actions, log_prob_actions, advantages, returns, optimizers, ppo_steps, ppo_clip):
    
    total_policy_loss = 0 
    total_value_loss = 0
    
    advantages = advantages.detach()
    log_prob_actions = log_prob_actions.detach()
    actions = actions.detach()
    
    for _ in range(ppo_steps):
                
        #get new log prob of actions for all input states
        action_pred, value_pred = networks[0](states), networks[1](states)
        value_pred = value_pred.squeeze(-1)
        action_prob = F.softmax(action_pred, dim = -1)
        dist = distributions.Categorical(action_prob)
        
        #new log prob using old actions
        new_log_prob_actions = dist.log_prob(actions)
        
        policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
                
        policy_loss_1 = policy_ratio * advantages
        policy_loss_2 = torch.clamp(policy_ratio, min = 1.0 - ppo_clip, max = 1.0 + ppo_clip) * advantages
        
        policy_loss = - torch.min(policy_loss_1, policy_loss_2).sum()
        
        value_loss = F.smooth_l1_loss(returns, value_pred).sum()
    
        [optm.zero_grad() for optm in optimizers]

        policy_loss.backward()
        value_loss.backward()

        [optm.step() for optm in optimizers]
    
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
    
    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps


def calculate_returns(rewards, discount_factor, normalize = True):
    
    returns = []
    R = 0
    
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
        
    returns = torch.tensor(returns)
    
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
        
    return returns

def calculate_advantages(returns, values, normalize = True):
    
    advantages = returns - values
    
    if normalize:
        
        advantages = (advantages - advantages.mean()) / advantages.std()
        
    return advantages

# %%
class PPO:
  def __init__(self, env_maker, batch_size=1, lr=0.01, gamma=0.99, clip=0.2, epochs=1000, n_ppo_updates=5, max_steps=1000, print_every=10):
    # Initialize hyperparameters
    self.batch_size = batch_size
    self.lr = lr
    self.gamma = gamma
    self.clip = clip
    self.epochs = epochs
    self.n_ppo_updates = n_ppo_updates
    self.max_steps = max_steps
    self.print_every = print_every
  
    # Initialize environment
    self.env = env_maker()

    # Initialize actor and critic models
    self.actor_model = MLP(in_dim=4, hidden_dim=64, out_dim=2)
    self.critic_model = MLP(in_dim=4, hidden_dim=64, out_dim=1)

    self.actor_model.apply(init_weights)
    self.critic_model.apply(init_weights)

    # Initialize optimizers for actor and critic
    self.actor_optim = Adam(self.actor_model.parameters(), lr=self.lr)
    self.critic_optim = Adam(self.critic_model.parameters(), lr=self.lr)
          
    # Initialize memory bank
    self.memory_bank = MemoryBank()

  def V(self, state, critic:Callable=None):
    critic = self.critic_model if critic is None else critic
    state = torch.cat(state) if isinstance(state, list) else state
    result = critic(state).squeeze() # [batch, 1] -> [batch]
    return result # [batch]
  
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
    advantages = advantages.detach()
    actions = actions.detach()
    log_action_probs = log_action_probs.detach()

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
      # critic_loss = nn.MSELoss()(V, rewards)
      critic_loss = F.smooth_l1_loss(V, rewards)

      # Updating the actor and critic
      self.actor_optim.zero_grad()
      actor_loss.backward(retain_graph=True)
      self.actor_optim.step()

      self.critic_optim.zero_grad()
      critic_loss.backward()
      self.critic_optim.step()

  def train(self):

    self.actor_model.train()
    self.critic_model.train()

    accumlated_reward = 0
    max_reward = 0
    past_episode_max_reward = False

    print("n_episode: ", self.epochs)
    for i in range(1, self.epochs+1):    
      # compute the trajectory usin the current policy
      batch_obs, batch_acts, batch_log_probs, rewards = batch_trajectory(self.env, self.actor_model, batch_size=self.batch_size, max_steps=self.max_steps)

      # compute the discounted rewards
      # batch_reward_otgs = discounted_rewards(rewards, discount_factor=self.gamma)
      batch_reward_otgs = calculate_returns(rewards, self.gamma)

      # compute the advantage
      V = self.V(batch_obs)
      A_k = batch_reward_otgs - V
      A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) # NORMALIZE

      # return a list of discounted rewards [steps]
      

      # PPO Update
      # self.update_PPO(batch_obs, batch_acts, batch_log_probs, batch_reward_otgs, A_k, self.clip, epochs=self.n_ppo_updates)
      update_policy((self.actor_model, self.critic_model), batch_obs, batch_acts, batch_log_probs, A_k, batch_reward_otgs, [self.actor_optim, self.critic_optim], self.n_ppo_updates, self.clip)

      trace_reward = sum(rewards) / self.batch_size
      accumlated_reward += trace_reward
      
      if trace_reward > max_reward:
        max_reward = trace_reward
        self.save_best_param()
        past_episode_max_reward = True
        
      if i % self.print_every == 0:
        if past_episode_max_reward:
          print(colored("rolling reward: " + str(accumlated_reward/self.print_every) + " max: " + str(max_reward), 'red'))
          past_episode_max_reward = False
        else:
          print("rolling reward: ", accumlated_reward/self.print_every)
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
