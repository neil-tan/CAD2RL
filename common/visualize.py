
from common.jupyter_animation import animate

def animate_policy(env, policy:callable, scale_factor:int=1, save_path:str=None):
  state, info = env.reset()
  image_seq = []

  done = False
  while not done:
    for i in range(scale_factor):
      state, reward, done, truncated, info = env.step(policy(state))
    image_seq.append(env.render())
  
  if save_path is not None:
    # save the sequence of images as a gif
    image_seq[0].save(save_path, save_all=True, append_images=image_seq[1:], duration=100, loop=0)

  return animate(image_seq)