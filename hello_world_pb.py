# %%
# https://colab.research.google.com/github/bulletphysics/bullet3/blob/656a1e74725933e947e5f64d465b62d6f9af683b/examples/pybullet/notebooks/HelloPyBullet.ipynb#scrollTo=tHb7uAveipon
# https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/cartpole_bullet.py
# https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24

import pybullet as p
from pb_utils import getCameraImage, print_joint_info, print_joint_states, print_body_states
from IPython.display import display
import pybullet_data


# %%
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# %%
# First, let's make sure we start with a fresh new simulation.
# Otherwise, we can keep adding objects by running this cell over again.
p.resetSimulation()

# Load our simulation floor plane at the origin (0, 0, 0).
p.loadURDF('plane.urdf')

# Load an R2D2 droid at the position at 0.5 meters height in the z-axis.
cartpole = p.loadURDF('cartpole.urdf', [0, 0, 0.5])

# We can check the number of bodies we have in the simulation.
p.getNumBodies()

# %%
# First let's define a class for the JointInfo.
from dataclasses import dataclass

@dataclass
class Joint:
  index: int
  name: str
  type: int
  gIndex: int
  uIndex: int
  flags: int
  damping: float
  friction: float
  lowerLimit: float
  upperLimit: float
  maxForce: float
  maxVelocity: float
  linkName: str
  axis: tuple
  parentFramePosition: tuple
  parentFrameOrientation: tuple
  parentIndex: int

  def __post_init__(self):
    self.name = str(self.name, 'utf-8')
    self.linkName = str(self.linkName, 'utf-8')

# Let's analyze the Cartpole!
print(f"Cartpole unique ID: {cartpole}")
print_joint_info(p, cartpole)


# %%
# Set the gravity to Earth's gravity.
p.setGravity(0, 0, -9.807)

p.setJointMotorControl2(cartpole, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

# Run the simulation for a fixed amount of steps.
for i in range(200):
    position, orientation = p.getBasePositionAndOrientation(cartpole)
    x, y, z = position

    p.setJointMotorControl2(cartpole, 0,
                                p.VELOCITY_CONTROL,
                                targetVelocity=-0.1)

    roll, pitch, yaw = p.getEulerFromQuaternion(orientation)
    print(f"{i:3}: x={x:0.10f}, y={y:0.10f}, z={z:0.10f}), roll={roll:0.10f}, pitch={pitch:0.10f}, yaw={yaw:0.10f}")
    p.stepSimulation()

# %%

rgba, depth, mask = getCameraImage(p, 320, 240)

print(f"rgba shape={rgba.size}")
display(rgba)
print(f"depth shape={depth.size}, as values from 0.0 (near) to 1.0 (far)")
display(depth)
print(f"mask shape={mask.size}, as unique values from 0 to N-1 entities, and -1 as None")
display(mask)

# %%
print_joint_states(p, cartpole, 0)
print_body_states(p, cartpole, 0)

# %%
def getPoleHeight(cartpole):
  pole_aabb_max = p.getAABB(cartpole, 1)[1]   
  cart_aabb_max = p.getAABB(cartpole, 0)[1]

  return pole_aabb_max[2] - cart_aabb_max[2]

print(f"Pole height: {getPoleHeight(cartpole)}")


# TODO: read getLinkStates in the PyBullet docs
# %%
