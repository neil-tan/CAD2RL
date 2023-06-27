# %%
import numpy as np
import copy
from termcolor import colored

from common.rl_common import batch_trajectory, discounted_rewards, log_action_prob
from typing import Iterable, Union, Callable, Tuple, List, Dict, Any
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# %%

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
class PPO:
  def __init__(self, env, policy_network:nn.Module, value_network:nn.Module, batch_size=1, lr=0.01, gamma=0.99, clip=0.2, epochs=500, n_ppo_updates=5, max_steps=1000, stop_at_reward=500, print_every=10):
    # Initialize hyperparameters
    self.batch_size = batch_size
    self.lr = lr
    self.gamma = gamma
    self.clip = clip
    self.epochs = epochs
    self.n_ppo_updates = n_ppo_updates
    self.max_steps = max_steps
    self.print_every = print_every
    self.stop_at_reward = stop_at_reward
  
    # Initialize environment
    self.env = env
    self.env.reset()

    # Initialize actor and critic models
    self.actor_model = policy_network
    self.critic_model = value_network

    # self.actor_model.apply(init_weights)
    # self.critic_model.apply(init_weights)

    # Initialize optimizers for actor and critic
    self.actor_optim = Adam(self.actor_model.parameters(), lr=self.lr)
    self.critic_optim = Adam(self.critic_model.parameters(), lr=self.lr)


  def V(self, state, critic:Callable=None):
    critic = self.critic_model if critic is None else critic
    state = torch.cat(state) if isinstance(state, list) else state
    result = critic(state).squeeze() # [batch, 1] -> [batch]
    return result # [batch]

  def update_PPO(self, observations, actions, log_action_probs, rewards, advantages, clip, epochs):
    advantages = advantages.detach()
    actions = actions.detach()
    log_action_probs = log_action_probs.detach()

    for _ in range(epochs):
      # estimate the value of the current state
      V = self.V(observations)

      # estimate the log probability of the past actions given the current policy
      new_action_log_probs = log_action_prob(self.actor_model, observations, actions)

      ratio = torch.exp(new_action_log_probs - log_action_probs) # [batch]

      # surrogate losses
      surr1 = ratio * advantages
      surr2 = torch.clamp(ratio, 1-clip, 1+clip) * advantages

      # actor loss
      actor_loss = -torch.min(surr1, surr2).mean()

      # critic loss
      critic_loss = nn.MSELoss()(V, rewards)

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
      batch_reward_otgs = discounted_rewards(rewards, discount_factor=self.gamma)

      # compute the advantage
      V = self.V(batch_obs)
      A_k = batch_reward_otgs - V
      A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) # NORMALIZE

      # return a list of discounted rewards [steps]
      

      # PPO Update
      self.update_PPO(batch_obs, batch_acts, batch_log_probs, batch_reward_otgs, A_k, self.clip, epochs=self.n_ppo_updates)

      trace_reward = sum(rewards).item() / self.batch_size
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
      
      if trace_reward >= self.stop_at_reward:
        print("Reached reward: ", self.stop_at_reward)
        break

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

