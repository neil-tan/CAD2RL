
from torch.utils.data import Dataset
import torch
import random

class MemoryBank(Dataset):
  def __init__(self, capacity:int=200000):
    super().__init__()
    self.capacity = capacity
    self.memory = []

  def __getitem__(self, index):
    return self.memory[index]
  
  def __iter__(self):
    for sample in self.memory:
      yield sample
  
  def __len__(self):
    return len(self.memory)
  
  def sample(self, batch_size):
    batch = random.sample(self.memory, batch_size)
    state, action, reward, new_state = [], [], [], []
    
    for sample in batch:
      state.append(sample.state)
      action.append(sample.action)
      reward.append(sample.reward)
      new_state.append(sample.new_state)
    
    state = torch.tensor(state).detach()
    action = torch.tensor(action, dtype=torch.int64).detach()
    reward = torch.tensor(reward).detach()
    new_state = torch.tensor(new_state).detach()
    
    return state, action, reward, new_state

  def add(self, item):
    is_full = len(self.memory) >= self.capacity
    if is_full:
      self.memory.pop(0)
    self.memory.append(item)
    return is_full

  def clear_memory(self):
    self.memory.clear()
