import asyncio
import concurrent.futures
from collections import namedtuple
import numpy as np
from .memory_bank import MemoryBank
from typing import Iterable, Union, Callable, Tuple, List, Dict, Any
from .eventloop_pool import EventLoopPool

class Agent:
  def __init__(self, env_maker:callable, q_function:asyncio.coroutine, epsilon=0.2):
    self.env = env_maker()
    self.state, info = self.env.reset()
    self.q_function = q_function
    self.epsilon = epsilon
    self.Q_Element = namedtuple('Q_Entry', ['state', 'action', 'reward', 'new_state'])

  async def step(self):
    if np.random.random() < self.epsilon:
      action = self.env.action_space.sample()
    else:
      action = self.q_function(self.state)
    # new_state, reward, done, truncated, info
    done, truncated, info = False, False, None
    while not done and not truncated:
      new_state, reward, done, truncated, info = self.env.step(action)
      yield self.state, action, reward, new_state
      self.state = new_state
  
  async def run(self) -> list:
    self.state, info = self.env.reset()
    trace = []
    async for state, action, reward, new_state in self.step():
      q_element = self.Q_Element(state, action, reward, new_state)
      trace.append(q_element)
    
    return trace


class AgentAnimator:
  def __init__(self, env_maker:callable, num_agents:int, q_function:asyncio.coroutine, threads:int=16):
    self.loop = asyncio.get_event_loop()
    self.num_agents = num_agents
    self.agents = []
    self.executor = EventLoopPool(num_workers=threads)
    self.q_function = q_function

    for _ in range(num_agents):
      self.agents.append(Agent(env_maker, self.q_function))

  
  # returns average length of traces
  def fill(self, memory_bank:MemoryBank, num_runs:int=0):
    average_length = 0

    if num_runs == 0:
      assert num_runs <= memory_bank.capacity, "num_runs must be less than or equal to memory_size"
      num_runs = memory_bank.capacity
    else:
      assert num_runs >= 0, "num_runs must be greater than or equal to 0"

    num_batches = num_runs // self.num_agents
    remainder_runs = num_runs % self.num_agents

    def traces_to_memory_bank(traces):
      nonlocal average_length
      for trace in traces:
        average_length += len(trace)
        for q_element in trace:
          memory_bank.add(q_element)
    

    for _ in range(num_batches):
      traces = self.run_all()
      traces_to_memory_bank(traces)
    
    if remainder_runs > 0:
      traces = self.run_all(self.agents[:remainder_runs])
      traces_to_memory_bank(traces)

    average_length /= num_runs
    return average_length

  def run_all(self, agents:List[Agent]=None):
    if agents is None:
      agents = self.agents

    coros = [agent.run() for agent in agents]
    result = self.executor.submit(coros)
    return result