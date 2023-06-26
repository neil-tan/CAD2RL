# %%
# import jax
# import jax.numpy as jnp
import numpy as np

from collections import namedtuple
import copy
from termcolor import colored

import gymnasium as gym

from common.memory_bank import MemoryBank
from common.agents import AgentAnimator, T_Element
from common.rl_common import batch_trajectory, discounted_rewards, log_action_prob
from typing import Iterable, Union, Callable, Tuple, List, Dict, Any
import time
import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from common.ppo import PPO
from common.visualize import animate_policy

# %%
env = gym.make('CartPole-v1', render_mode="rgb_array")
# %%
animate_policy(env, lambda state: np.random.randint(0,1))

# %%
test_obj = PPO(env=env)
test_obj.train()

print("Training finished")
# %%
animate_policy(env, lambda state: test_obj.select_action(state))
# %%
