import pytest
from ..common.memory_bank import MemoryBank
from ..common.agents import Q_Element

def test_add(capsys):
  memory_bank = MemoryBank(10)
  for i in range(0,10):
    memory_bank.add(i)
    assert len(memory_bank.memory) == i+1
  
  for i in range(0,10):
    assert memory_bank[i] == i

def test_add_overflow(capsys):
  memory_bank = MemoryBank(10)
  for i in range(0,20):
    memory_bank.add(i)
    if i < 10:
      assert len(memory_bank.memory) == i+1
    else:
      assert len(memory_bank.memory) == 10
  
  for i in range(0,10):
    assert memory_bank[i] == i+10

def test_clear_memory(capsys):
  memory_bank = MemoryBank(10)
  for i in range(0,10):
    memory_bank.add(i)
    assert len(memory_bank.memory) == i+1
  
  memory_bank.clear_memory()
  assert len(memory_bank.memory) == 0

def test_iter(capsys):
  memory_bank = MemoryBank(10)
  for i in range(0,10):
    memory_bank.add(i)
    assert len(memory_bank.memory) == i+1
  
  for i, item in enumerate(memory_bank):
    assert item == i

def test_sample(capsys):
  memory_bank = MemoryBank(10)
  for i in range(0,10):
    memory_bank.add(Q_Element(i,i,i,i))
    assert len(memory_bank.memory) == i+1
  
  state, action, reward, new_state = memory_bank.sample(5)
  assert len(state) == 5
  assert len(action) == 5
  assert len(reward) == 5
  assert len(new_state) == 5