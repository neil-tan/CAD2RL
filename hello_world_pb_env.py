# %%
import numpy as np
from PIL import Image

import gymnasium as gym
from gymnasium import spaces


import pybullet as p
# from pb_utils import getCameraImage, print_joint_info, print_joint_states, print_body_states
from IPython.display import display
import pybullet_data

# %%
class CartPolePyBulletEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode

        self.observation_space = spaces.Box(np.array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38]),
                                            np.array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38]), (4,), np.float32)
        
        self.action_space = spaces.Discrete(1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.physID = p.connect(p.DIRECT)
        
        p.loadURDF('plane.urdf')
        self.cartpole = p.loadURDF('cartpole.urdf', [0, 0, 0.5])
        p.setGravity(0, 0, -9.807, physicsClientId=self.physID)
        self.reset()

    def reset(self, seed=None, options=None):
        self._seed = seed
        self._options = options

        p.resetSimulation(physicsClientId=self.physID)
        p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=0, physicsClientId=self.physID)

        # get obs
        # get info
        # reset render mode?
    

        return self._get_observation(), info

    def step(self, action):
        position, orientation = p.getBasePositionAndOrientation(self.cartpole)
        x, y, z = position

        p.setJointMotorControl2(self.cartpole, 0,
                                p.VELOCITY_CONTROL,
                                targetVelocity=1 if action == 1 else -1,)

        p.stepSimulation()

        # construct and return obsveration, reward, check termination, info
    
    def getPoleHeight(self):
        pole_aabb_max = p.getAABB(self.cartpole, 1, physicsClientId=self.physID)[1]   
        cart_aabb_max = p.getAABB(self.cartpole, 0, physicsClientId=self.physID)[1]

        return pole_aabb_max[2] - cart_aabb_max[2]
    
    def getCartPosition(self):
        link_state = p.getLinkState(self.cartpole, 0, computeLinkVelocity=1, physicsClientId=self.physID)
        return link_state[0]
    
    def render(self, width=320, height=240):
        if self.render_mode == "rgb_array":
            img_arr = p.getCameraImage(
                                        width,
                                        height,
                                        viewMatrix=p.computeViewMatrixFromYawPitchRoll(
                                            cameraTargetPosition=[0, 0, 0.5],
                                            distance=2.5,
                                            yaw=0,
                                            pitch=-15,
                                            roll=0,
                                            upAxisIndex=2,
                                        ),
                                        projectionMatrix=p.computeProjectionMatrixFOV(
                                            fov=60,
                                            aspect=width/height,
                                            nearVal=0.01,
                                            farVal=100,
                                        ),
                                        shadow=True,
                                        lightDirection=[1, 1, 1],
                                        physicsClientId=self.physID,
                                    )
            w, h, rgba, depth, mask = img_arr
            rgba_image = Image.fromarray(rgba.reshape(h, w, 4).astype(np.uint8))
            return rgba_image
    
    def close(self):
        p.disconnect(physicsClientId=self.physID)

# TODO:
# https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#registering-envs
# step and reset still needs to be implemented