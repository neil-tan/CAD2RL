from ..common.agents import Agent
import numpy as np
import gym
import asyncio

def q_function(state):
  return np.random.randint(0, 2)

def test_agent():
  env_maker = lambda: gym.make('CartPole-v1', render_mode="rgb_array")
  agent = Agent(env_maker, q_function)
  loop = asyncio.get_event_loop()
  trace = loop.run_until_complete(agent.run())
  assert len(trace) > 0

def test_agent_repeat(capsys):
  env_maker = lambda: gym.make('CartPole-v1', render_mode="rgb_array")
  agent = Agent(env_maker, q_function)
  loop = asyncio.get_event_loop()
  trace_lens = []
  for _ in range(10):
    trace = loop.run_until_complete(agent.run())
    trace_lens.append(len(trace))
    
  for trace_len in trace_lens:
    assert trace_len > 0
  
  print("lens: ", trace_lens, end='')
