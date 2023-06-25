# %%
import gymnasium as gym
import hello_world_pb_env

# %%
custom_environment = gym.envs.registration.EnvSpec(id='my_env/pybullet_cartpole-v1', 
                                                   entry_point=hello_world_pb_env.CartPolePyBulletEnv,
                                                   reward_threshold=500, 
                                                   )

# %%
env = gym.make(custom_environment)

# %%
print("hello")

# %%
