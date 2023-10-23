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

    def __init__(self, render_mode=None, targetVelocity=0.1, max_force=100, step_scaler:int=1):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.x_threshold = 2.4
        self.theta_threshold_degrees = 12
        self.theta_threshold_radians = self.theta_threshold_degrees * 2 * np.pi / 360 # 0.209
        self.targetVelocity = targetVelocity
        self.max_force = max_force
        self.step_scaler = step_scaler

        self.done = False

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
        angular_velocity = link_state[7][1]
        # assuming the pole is not rotating around the x and y axis
        angle = p.getAxisAngleFromQuaternion(link_state[5], physicsClientId=self.physID)[1]
        return position, angular_velocity, angle

    def _should_terminate(self, position, angle):
        return (
            position < -self.x_threshold
            or position > self.x_threshold
            or angle < -self.theta_threshold_radians
            or angle > self.theta_threshold_radians
        )

    def _step_simulation(self):
        for _ in range(self.step_scaler):
            p.stepSimulation(physicsClientId=self.physID)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._seed = seed
        self._options = options
        self.current_steps_count = 0
        self.done = False

        p.resetSimulation(physicsClientId=self.physID)

        p.loadURDF('plane.urdf', physicsClientId=self.physID)
        self.cartpole = p.loadURDF('cartpole.urdf', [0, 0, 0.5], physicsClientId=self.physID)
        p.setGravity(0, 0, -9.807, physicsClientId=self.physID)

        p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=0, physicsClientId=self.physID)

        # randomize initial condition
        random_condition_gen = lambda : np.random.uniform(low=-0.05, high=0.05)
        init_cart_position = random_condition_gen()
        init_cart_velocity = random_condition_gen()
        init_pole_angle = random_condition_gen()
        init_pole_angular_velocity = random_condition_gen()

        p.resetJointState(self.cartpole, 0, init_cart_position, targetVelocity=init_cart_velocity, physicsClientId=self.physID)
        p.resetJointState(self.cartpole, 1, init_pole_angle, targetVelocity=init_pole_angular_velocity, physicsClientId=self.physID)

        self._step_simulation()

        return self._get_obs(), self._get_info()

    def step(self, action):
        position, orientation = p.getBasePositionAndOrientation(self.cartpole)
        x, y, z = position

        p.setJointMotorControl2(self.cartpole, 0,
                                p.VELOCITY_CONTROL,
                                targetVelocity=self.targetVelocity if action == 1 else -self.targetVelocity,
                                force=self.max_force,
                                physicsClientId=self.physID)

        self._step_simulation()

        observation = self._get_obs()
        cart_position, cart_velocity, pole_angle, pole_velocity = observation
        
        reward = 0
        if not self.done:
            self.done = self._should_terminate(cart_position, pole_angle)
            reward = 1.0

        return observation, reward, self.done, False, self._get_info()

    
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
