import asyncio
import concurrent.futures
from collections import namedtuple
import numpy as np
from .memory_bank import MemoryBank
from typing import Iterable, Union, Callable, Tuple, List, Dict, Any
from .eventloop_pool import EventLoopPool

T_Element = namedtuple('Q_Entry', ['state', 'action', 'reward', 'new_state', 'discounted_reward'])
class Agent:
  def __init__(self, env_maker:callable, q_function:asyncio.coroutine, epsilon=0.7):
    self.env = env_maker()
    self.state, info = self.env.reset()
    self.q_function = q_function
    self.epsilon = epsilon
    self.T_Element = T_Element

  async def step(self):
    if np.random.random() < self.epsilon:
      action = self.env.action_space.sample()
    else:
      action = self.q_function(self.state)
    # new_state, reward, done, truncated, info
    done, truncated, info = False, False, None
    while not done and not truncated:
      new_state, reward, done, truncated, info = self.env.step(action)
      # FIXME:
      if done and not truncated:
        reward = -1
      yield self.state, action, reward, new_state
      self.state = new_state
    
  def compute_discounted_reward(self, trace:list[T_Element], gamma=0.9):
    # at least one element
    if len(trace) > 0:
      trace[-1] = trace[-1]._replace(discounted_reward=trace[-1].reward)

    # has at least two elements
    if len(trace) > 1:
      trace[-2] = trace[-2]._replace(discounted_reward=(trace[-2].reward + gamma * trace[-1].discounted_reward))
    
    if len(trace) > 2:
      for i in range(len(trace)-3, -1, -1):
        trace[i] = trace[i]._replace(discounted_reward=(trace[i].reward + gamma * trace[i+1].discounted_reward))
  
  async def run(self, epsilon=None) -> list[T_Element]:
    if epsilon is not None:
      self.set_epsilon(epsilon)

    self.state, info = self.env.reset()
    trace = []
    async for state, action, reward, new_state in self.step():
      t_element = self.T_Element(state, action, reward, new_state, 0)
      trace.append(t_element)

    self.compute_discounted_reward(trace)
    
    return trace
  
  def set_epsilon(self, epsilon):
    self.epsilon = epsilon


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
  def fill(self, memory_bank:MemoryBank, num_runs:int=0, epsilon=0.7):
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
        for t_element in trace:
          memory_bank.add(t_element)
    

    for _ in range(num_batches):
      traces = self.run_all(epsilon=epsilon)
      traces_to_memory_bank(traces)
    
    if remainder_runs > 0:
      traces = self.run_all(self.agents[:remainder_runs], epsilon=epsilon)
      traces_to_memory_bank(traces)

    average_length /= num_runs
    return average_length

  def run_all(self, agents:List[Agent]=None, epsilon=0.7) -> List[T_Element]:
    if agents is None:
      agents = self.agents

    coros = [agent.run(epsilon=epsilon) for agent in agents]
    result = self.executor.submit(coros)
    return result