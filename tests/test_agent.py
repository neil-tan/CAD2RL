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

def test_discount_rewards(capsys):
  env_maker = lambda: gym.make('CartPole-v1', render_mode="rgb_array")
  agent = Agent(env_maker, q_function)
  loop = asyncio.get_event_loop()

  trace = []
  retry = 0

  while len(trace) < 10:
    trace = loop.run_until_complete(agent.run())
    retry += 1
    if retry > 100:
      assert "retry too many times"
  
  for i in range(len(trace)-1,-1,-1):
    if i == len(trace)-1:
      assert trace[i].reward == trace[i].discounted_reward
      continue

    assert trace[i].reward + 0.9 * trace[i+1].discounted_reward == trace[i].discounted_reward
  
  assert len(trace) == 10