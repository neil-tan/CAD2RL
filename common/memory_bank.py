
from torch.utils.data import IterableDataset, Dataset

class MemoryBank(Dataset):
  def __init__(self, capacity:int=200000):
    super().__init__()
    self.capacity = capacity
    self.memory = []
    self.memory_pointer = 0

  def __getitem__(self, index):
    return self.memory[index]
  
  def __iter__(self):
    for sample in self.memory:
      yield sample
  
  def __len__(self):
    return len(self.memory)
  
  def add(self, item):
    is_full = len(self.memory) >= self.capacity
    if is_full:
      self.memory.pop(0)
    self.memory.append(item)
    return is_full

  def clear_memory(self):
    self.memory.clear()
