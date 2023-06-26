# %%
import numpy as np
import gymnasium as gym
import hello_world_pb_env
from common.ppo import PPO
from common.visualize import animate_policy
# %%
custom_environment_spec = gym.envs.registration.EnvSpec(id='my_env/pybullet_cartpole-v1', 
                                                   entry_point=hello_world_pb_env.CartPolePyBulletEnv,
                                                   reward_threshold=500, 
                                                   )
env = gym.make(custom_environment_spec, render_mode="rgb_array")
# %%
animate_policy(env, lambda state: np.random.randint(0,1))
# %%
test_obj = PPO(env=env)
test_obj.train()
# %%
print("Training finished.")

# %%
animate_policy(env, lambda state: test_obj.select_action(state))

# %%