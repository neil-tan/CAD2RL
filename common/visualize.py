
from common.jupyter_animation import animate

def animate_policy(env, policy:callable, scale_factor:int=1, save_path:str=None):
  state, info = env.reset()
  image_seq = []

  done = False
  while not done:
    for i in range(scale_factor):
      state, reward, done, truncated, info = env.step(policy(state))
    image_seq.append(env.render())

  return animate(image_seq, save_path=save_path)