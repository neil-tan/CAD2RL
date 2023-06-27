# %%
import numpy as np
import gymnasium as gym
import hello_world_pb_env
from common.ppo import PPO, MLP
from common.visualize import animate_policy

# %%
custom_environment_spec = gym.envs.registration.EnvSpec(id='my_env/pybullet_cartpole-v1', 
                                                   entry_point=hello_world_pb_env.CartPolePyBulletEnv,
                                                   reward_threshold=2000, 
                                                   max_episode_steps=2000,
                                                   )
env = gym.make(custom_environment_spec, render_mode="rgb_array", max_force=1000, targetVelocity=0.15)
# %%
animate_policy(env, lambda state: np.random.randint(0,1), scale_factor=4)
# %%

test_obj = PPO(env=env,
               policy_network=MLP(in_dim=4, hidden_dim=1024, out_dim=2),
               value_network=MLP(in_dim=4, hidden_dim=1024, out_dim=1),
               lr=0.01,
               gamma=0.99,
               epochs=2000,
               n_ppo_updates=10,
               max_steps=2000,
               stop_at_reward=2000,
               print_every=20,
               )
test_obj.train()
# %%
print("Training finished.")

# %%
animate_policy(env, lambda state: test_obj.select_action(state), scale_factor=4)

# %%
# TODO: Random state start