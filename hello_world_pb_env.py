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
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.max_episode_steps = 200

        self.observation_space = spaces.Box(np.array([-4.8000002e+00, -3.4028235e+38, -4.1887903e-01, -3.4028235e+38]),
                                            np.array([4.8000002e+00, 3.4028235e+38, 4.1887903e-01, 3.4028235e+38]), (4,), np.float32)
        
        self.action_space = spaces.Discrete(1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.physID = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.reset()

    def _get_obs(self):
        # Returns
        # Cart Position -4.8 ~ 4.8        
        # Cart Velocity -inf ~ inf
        # Pole Angle -0.418 ~ 0.418
        # Pole Velocity At Tip -inf ~ inf

        # assume only moves in x
        cart_position = p.getLinkState(self.cartpole, 0, computeLinkVelocity=1, physicsClientId=self.physID)[0][0]
        cart_velocity = p.getLinkState(self.cartpole, 0, computeLinkVelocity=1, physicsClientId=self.physID)[6][0]
        _, pole_angular_velocity, pole_angle = self._getPoleStates(self.cartpole)
        
        result = np.array([cart_position, cart_velocity, pole_angle, pole_angular_velocity], dtype=np.float32)
        return result

    def _get_info(self):
        return {}
        
    def _getPoleStates(self, cartpole):
        link_state = p.getLinkState(cartpole, 1, computeLinkVelocity=1, physicsClientId=self.physID)
        position = link_state[0]
        angular_velocity = link_state[7][0]
        # assuming the pole is not rotating around the x and y axis
        angle = p.getAxisAngleFromQuaternion(link_state[5], physicsClientId=self.physID)[0][1]
        return position, angular_velocity, angle

    def _should_terminate(self, position, angle):
        return (
            position < -self.x_threshold
            or position > self.x_threshold
            or angle < -self.theta_threshold_radians
            or angle > self.theta_threshold_radians
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._seed = seed
        self._options = options
        self.current_steps_count = 0

        p.resetSimulation(physicsClientId=self.physID)

        p.loadURDF('plane.urdf', physicsClientId=self.physID)
        self.cartpole = p.loadURDF('cartpole.urdf', [0, 0, 0.5], physicsClientId=self.physID)
        p.setGravity(0, 0, -9.807, physicsClientId=self.physID)

        p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=0, physicsClientId=self.physID)

        # get obs
        # get info
        # reset render mode?

        return self._get_obs(), self._get_info()

    def step(self, action):
        position, orientation = p.getBasePositionAndOrientation(self.cartpole)
        x, y, z = position

        p.setJointMotorControl2(self.cartpole, 0,
                                p.VELOCITY_CONTROL,
                                targetVelocity=1 if action == 1 else -1,)

        p.stepSimulation()

        observation = self._get_obs()
        cart_position, cart_velocity, pole_angle, pole_velocity = observation
        done = self._should_terminate(cart_position, pole_angle)
        reward = 1.0 if not done else 0.0

        return observation, reward, done, False, self._get_info()

    
    def getPoleHeight(self):
        pole_aabb_max = p.getAABB(self.cartpole, 1, physicsClientId=self.physID)[1]   
        cart_aabb_max = p.getAABB(self.cartpole, 0, physicsClientId=self.physID)[1]

        return pole_aabb_max[2] - cart_aabb_max[2]
    
    def getCartPosition(self):
        link_state = p.getLinkState(self.cartpole, 0, computeLinkVelocity=1, physicsClientId=self.physID)
        return link_state[0]
    
    def render(self, width=320, height=240):
        if self.render_mode is not None and self.render_mode != "rgb_array":
            raise NotImplementedError("Only rgb_array render mode is supported")

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
