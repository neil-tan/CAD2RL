from typing import Callable
import torch
import torch.nn.functional as F

def list_to_tensor(l:list)->torch.Tensor:
  # taking care of native python types
  if not hasattr(l[0], 'dim'):
    return torch.tensor(l)

  # scalar tensor
  if l[0].dim() == 0:
    return torch.tensor(l)

  # tensor contain a single vector
  if l[0].dim() == 1 and l[0].shape[-1] > 1:
    return torch.stack(l)

  return torch.cat(l)

# returns (action:tensor[int], log_prob_action:tensor[float]) from policy given state
def act(policy:Callable, state:torch.Tensor)->tuple[torch.tensor, torch.tensor]:
  assert state.dim() == 1, f"state must be a 1D tensor, got {state.dim()}D tensor"
  action_pred = policy(state) # [action_dims]
  action_prob = F.softmax(action_pred, dim=-1) # [action_dims]
  dist = torch.distributions.Categorical(action_prob) # action_dims
  action = dist.sample() # Catagorical -> []
  log_prob_action = dist.log_prob(action) # log(action_prob[action]) -> [] # probability of action taken

  # if log_prob_action.dim() == 0:
  #   log_prob_action = log_prob_action.unsqueeze(0)

  return action, log_prob_action

# compute the trajectory usin the current policy
# returns a list of rewards and a 1D-tensor of log probabilities of actions
def sample_trajectory(env, policy:Callable, max_steps=1000):
  state, info = env.reset()
  done = False
  truncated = False
  rewards = []
  states = []
  actions = []
  log_prob_actions = []
  episode_reward = 0

  while not done and not truncated:
    state = torch.tensor(state) # num_states [4]
    action, log_prob_action = act(policy, state) # [int], [float]

    states.append(state) # states -> list:[current_num_step + 1], state -> tensor[4]
    actions.append(action) # actions -> list:[current_num_step + 1], action -> int

    state, reward, done, truncated, info = env.step(action.item())
    rewards.append(reward) # rewards -> list:[current_num_step + 1], reward -> float
    log_prob_actions.append(log_prob_action) # [current_num_step + 1, num_action  

    episode_reward += reward
    if len(rewards) > max_steps:
      break
      
  # Convert Trojectory to Tensors
  # Stacking list of [current_num_step + 1][float], to [current_num_step + 1, 1] -> [sampled_distribution_elements]
  log_prob_actions = list_to_tensor(log_prob_actions)
  states = list_to_tensor(states)
  actions = list_to_tensor(actions)
  rewards = list_to_tensor(rewards)

  return states, actions, log_prob_actions, rewards

def discounted_rewards(rewards, discount_factor, normalize = True):
    """ Calculate discounted returns from rewards
    Args:
      rewards list[float]: list of rewards for each step
      discount_factor float: discount factor
      normalize bool: normalize returns

    Returns:
      list[float]: list of discounted returns
    """

    returns = []
    R = 0
    
    for r in reversed(rewards):
        R = r + R * discount_factor
        returns.insert(0, R)
        
    returns = torch.tensor(returns)
    
    if normalize:
        returns = (returns - returns.mean()) / returns.std()
        
    return returns

# returns a batch of trajectories in flattened form
def batch_trajectory(env, policy:Callable, batch_size:int=1, max_steps=1000, individual_trajectories=False):
  observations = []
  actions = []
  log_prob_actions = []
  rewards = []

  for _ in range(batch_size):
    obs, act, log_prob_act, reward = sample_trajectory(env, policy, max_steps=max_steps)
    observations.append(obs)
    actions.append(act)
    log_prob_actions.append(log_prob_act)
    rewards.append(reward)

  combinator_function = torch.stack if individual_trajectories else torch.cat

  observations = combinator_function(observations)
  actions = combinator_function(actions)
  log_prob_actions = combinator_function(log_prob_actions)
  rewards = combinator_function(rewards)

  return observations, actions, log_prob_actions, rewards

def log_action_prob(policy:Callable, state, action):
  state = torch.cat(state) if isinstance(state, list) else state
  action = torch.cat(action) if isinstance(action, list) else action

  action_pred = policy(state) # num_actions [batch, action dims] -> [1, 2]
  action_prob = F.softmax(action_pred, dim=-1) # num_actions [batch, 2]
  dist = torch.distributions.Categorical(action_prob) # [batch, 2]
  log_prob_action = dist.log_prob(action) # log(action_prob[action]) -> [batch] # probability of action taken
  
  return log_prob_action