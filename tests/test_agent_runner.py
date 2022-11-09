from ..common.agents import Agent, AgentAnimator
from ..common.memory_bank import MemoryBank
import numpy as np
import gym
import asyncio

def q_function(state):
  return np.random.randint(0, 2)


def test_agent_animator_run_all():
  env_maker = lambda: gym.make('CartPole-v1', render_mode="rgb_array")
  agent_runner = AgentAnimator(env_maker, num_agents=10, q_function=q_function, threads=4)
  traces = agent_runner.run_all(agent_runner.agents)
  assert len(traces) == 10
  for trace in traces:
    assert len(trace) > 0
    for q_element in trace:
      assert len(q_element) == 4

def test_agent_animator():
  memory_bank = MemoryBank(capacity=1000)
  env_maker = lambda: gym.make('CartPole-v1', render_mode="rgb_array")
  agent_runner = AgentAnimator(env_maker, num_agents=10, q_function=q_function, threads=4)
  agent_runner.fill(memory_bank)
  assert len(memory_bank) == memory_bank.capacity