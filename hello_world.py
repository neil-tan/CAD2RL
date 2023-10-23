# %%
import numpy as np
import gymnasium as gym
import custom_envs.cartpole_pb_env as hello_world_pb_env
from common.ppo import PPO, MLP
from common.visualize import animate_policy
import matplotlib.pyplot as plt

# %%
custom_environment_spec = gym.envs.registration.EnvSpec(id='my_env/pybullet_cartpole-v1', 
                                                   entry_point=hello_world_pb_env.CartPolePyBulletEnv,
                                                   reward_threshold=2000, 
                                                   max_episode_steps=2000,
                                                   )
env = gym.make(custom_environment_spec, render_mode="rgb_array", max_force=1000, targetVelocity=1)
# %%
animate_policy(env, lambda state: np.random.randint(0,1), scale_factor=4)
# %%
test_obj = PPO(env=env,
               policy_network=MLP(in_dim=4, hidden_dim=256, out_dim=2),
               value_network=MLP(in_dim=4, hidden_dim=256, out_dim=1),
               batch_size=1,
               lr=0.01,
               gamma=0.99,
               epochs=300,
               n_ppo_updates=5,
               max_steps=2000,
               stop_at_reward=None,
               print_every=20,
               )
test_obj.train()
# %%
print("Training finished.")

# %%
animate_policy(env, lambda state: test_obj.select_action(state), scale_factor=4, save_path="test.gif")

# %%