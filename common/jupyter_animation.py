
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from typing import Iterable, Union, Callable, Tuple, List, Dict, Any

def animate(image_seq: Iterable, axes=None, close_fig=True, save_path=None):
  from matplotlib import rc
  rc('animation', html='html5')

  fig = plt.figure() # get current figure
  if axes is None:
    fig, axes = plt.subplots(1,1)

  graphical_element = axes.imshow(image_seq[0])

  def animate_frame(i):
    graphical_element.set_data(image_seq[i])
    return graphical_element,
  animation_handler = animation.FuncAnimation(fig, animate_frame, frames=len(image_seq), interval=50)

  if close_fig:
    plt.close(fig)

  if save_path is not None:
    # save the sequence of images as a gif
    image_seq[0].save(save_path, save_all=True, append_images=image_seq[1:], duration=50, loop=0)
    
  return animation_handler

# Example:
# a = animate(image_seq)
# print(type(a))
# a

#########
# support spanning: image_seq = [[img1, img2, (img3, 1, 2)], [img4, img5, None]]
# a = animation_table([[image_seq, (image_seq, 1, 2)], [image_seq, None]], (2,2))
# image_seq : iterable of animation data
# animation_data: list of images, or, (list of images, col_span, row_span)
# TODO: https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array

from matplotlib.pyplot import grid

def animation_table(image_seq: Iterable, grids: Iterable[int], min_num_frames=0, close_fig=True):
  from matplotlib import rc
  rc('animation', html='html5')

  def is_meta_image(image):
    return isinstance(image, tuple) and len(image) == 3 and isinstance(image[1], int) and isinstance(image[2], int)

  graphical_element_mapping = dict()
  max_num_frames = min_num_frames
  # return number of frames
  def register_animation(animation_data, row, col) -> int:
    if animation_data is None:
      return 0
    col_span = 1
    row_span = 1
    if is_meta_image(animation_data):
      animation_data, col_span, row_span = animation_data
    ax = plt.subplot2grid((grids[0], grids[1]), (row, col), row_span, col_span)
    graphical_element = ax.imshow(animation_data[0]) # first image
    graphical_element_mapping[graphical_element] = animation_data
    
    return len(animation_data)

  assert len(grids) == 2
  assert len(image_seq) == grids[1], "The row number of image_seq should be equal to the row number of grids"
  for i, image_row in enumerate(image_seq):
    assert len(image_row) == grids[0], "The column number of image_seq should be equal to the column number of grids"
    for j, animation_data in enumerate(image_row):
      num_frames = register_animation(animation_data, i, j)
      max_num_frames = max(max_num_frames, num_frames)

  fig = plt.gcf()

  def animate_frame(i):
    for graphical_element, animation_data in graphical_element_mapping.items():
      graphical_element.set_data(animation_data[min(i, len(animation_data)-1)])
    return tuple(graphical_element_mapping.keys())
  animation_handler = animation.FuncAnimation(fig, animate_frame, frames=max_num_frames, interval=50)

  if close_fig:
    plt.close(fig)
    
  return animation_handler

# a = animation_table([[image_seq, (image_seq, 1, 2)], [image_seq, None]], (2,2))
# a