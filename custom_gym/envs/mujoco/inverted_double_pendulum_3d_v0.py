import os
from typing import Dict, Union

import numpy as np
from scipy.spatial.transform import Rotation

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 1.0)),
}

class InvertedDoublePendulum3DEnv(MujocoEnv, utils.EzPickle):
    """
    ## Description
    This environment is a 3D extension of the Inverted Double Pendulum environment.
    The goal is to balance two poles on a cart that can move linearly in the x-y plane.

    ## Action Space
    The agent takes a 2-element vector for actions.
    The action space is a continuous `(action_x, action_y)` in `[-10, 10]^2`, where `action_x` and `action_y`
    represent the numerical force applied to the cart in the x and y directions respectively.

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit) |
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-------------|
    | 0   | Force applied on the cart in x-direction | -10 | 10  | slider_x | slide | Force (N) |
    | 1   | Force applied on the cart in y-direction | -10 | 10  | slider_y | slide | Force (N) |

    ## Observation Space
    The observation space consists of positional and velocity values of the cart and both poles.
    The poles' orientations are represented using quaternions.

    | Num | Observation                                   | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
    | --- | --------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------- |
    | 0   | position of the cart in the x-direction       | -Inf | Inf | slider_x | slide | position (m) |
    | 1   | position of the cart in the y-direction       | -Inf | Inf | slider_y | slide | position (m) |
    | 2   | w-component of the first pole's quaternion    | -Inf | Inf | ball1 (quaternion) | ball | quaternion component |
    | 3   | x-component of the first pole's quaternion    | -Inf | Inf | ball1 (quaternion) | ball | quaternion component |
    | 4   | y-component of the first pole's quaternion    | -Inf | Inf | ball1 (quaternion) | ball | quaternion component |
    | 5   | z-component of the first pole's quaternion    | -Inf | Inf | ball1 (quaternion) | ball | quaternion component |
    | 6   | w-component of the second pole's quaternion   | -Inf | Inf | ball2 (quaternion) | ball | quaternion component |
    | 7   | x-component of the second pole's quaternion   | -Inf | Inf | ball2 (quaternion) | ball | quaternion component |
    | 8   | y-component of the second pole's quaternion   | -Inf | Inf | ball2 (quaternion) | ball | quaternion component |
    | 9   | z-component of the second pole's quaternion   | -Inf | Inf | ball2 (quaternion) | ball | quaternion component |
    | 10  | velocity of the cart in the x-direction       | -Inf | Inf | slider_x | slide | velocity (m/s) |
    | 11  | velocity of the cart in the y-direction       | -Inf | Inf | slider_y | slide | velocity (m/s) |
    | 12  | angular velocity of the first pole about x-axis  | -Inf | Inf | ball1 | ball | angular velocity (rad/s) |
    | 13  | angular velocity of the first pole about y-axis  | -Inf | Inf | ball1 | ball | angular velocity (rad/s) |
    | 14  | angular velocity of the first pole about z-axis  | -Inf | Inf | ball1 | ball | angular velocity (rad/s) |
    | 15  | angular velocity of the second pole about x-axis | -Inf | Inf | ball2 | ball | angular velocity (rad/s) |
    | 16  | angular velocity of the second pole about y-axis | -Inf | Inf | ball2 | ball | angular velocity (rad/s) |
    | 17  | angular velocity of the second pole about z-axis | -Inf | Inf | ball2 | ball | angular velocity (rad/s) |

    ## Rewards
    The reward is computed using an exponential function for the cart's distance from the origin,
    cosine of the angles of both poles, and penalties for angular velocities and actions.

    ## Starting State
    The initial position and velocity of the cart and both poles are randomly sampled from a uniform distribution
    with range [-reset_noise_scale, reset_noise_scale].

    ## Episode End
    The episode ends if any of the following occurs:
    1. Any of the state space values are no longer finite.
    2. The absolute value of the angle of the lower pole (angle1) with respect to vertical exceeds 0.4 radians.
    3. The absolute value of the angle of the upper pole (angle2) with respect to vertical exceeds 0.6 radians.
    4. The cart moves more than 10 units away from the origin in either the x or y direction.
    5. The maximum number of steps is reached.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = None,
        frame_skip: int = 2,
        default_camera_config: Dict[str, Union[float, int, np.ndarray]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 0.01,
        max_episode_steps: int = 30000,
        **kwargs,
    ):
        if xml_file is None:
            xml_file = os.path.join(os.path.dirname(__file__), "assets", "inverted_double_pendulum_3d.xml")

        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64)

        self._reset_noise_scale = reset_noise_scale
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def compute_reward(self, observation, action, angle1, angle2):
        # Angle reward (unchanged)
        angle_reward = 2.0 * (np.cos(angle1)**2 + np.cos(angle2)**2 - 1)
        
        # Enhanced position reward
        cart_x, cart_y = observation[0], observation[1]
        distance_from_origin = np.sqrt(cart_x**2 + cart_y**2)
        position_reward = 5.0 * np.exp(-10 * distance_from_origin)  # Increased weight and sharpness
        
        # Additional reward for being very close to the center
        center_bonus = 2.0 if distance_from_origin < 0.1 else 0.0
        
        # Velocity penalty
        cart_vx, cart_vy = observation[10], observation[11]
        velocity = np.sqrt(cart_vx**2 + cart_vy**2)
        velocity_penalty = -0.1 * velocity
        
        # Angular velocity penalty (unchanged)
        angular_velocity = np.linalg.norm(observation[12:18])
        angular_velocity_penalty = -0.05 * angular_velocity
        
        # Action penalty (unchanged)
        action_penalty = -0.005 * np.sum(np.square(action))
        
        # Total reward calculation
        reward = angle_reward + position_reward + center_bonus + velocity_penalty + angular_velocity_penalty + action_penalty
        
        return reward, {
            "angle_reward": angle_reward,
            "position_reward": position_reward,
            "center_bonus": center_bonus,
            "velocity_penalty": velocity_penalty,
            "angular_velocity_penalty": angular_velocity_penalty,
            "action_penalty": action_penalty
        }

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        
        observation = self._get_obs()
        
        # Obtaining quaternion values from the observation for both poles
        quat1 = observation[2:6]
        quat2 = observation[6:10]
        
        # Transformation for "scipy" (x, y, z, w)
        quat1_xyzw = np.roll(quat1, -1)
        quat2_xyzw = np.roll(quat2, -1)
        
        # Quaternion to Euler
        r1 = Rotation.from_quat(quat1_xyzw)
        r2 = Rotation.from_quat(quat2_xyzw)
        
        # Radian Conversion
        angle1 = r1.magnitude()
        angle2 = r2.magnitude()
        
        # Reward Calculation
        reward, reward_info = self.compute_reward(observation, action, angle1, angle2)
        
        # Terminate if angle1 > 0.4 radian or angle2 > 0.6 or cart goes beyond spaces
        terminated = bool(
            not np.isfinite(observation).all() or 
            (angle1 > 0.4) or (angle2 > 0.6) or
            (np.abs(observation[0]) > 10) or 
            (np.abs(observation[1]) > 10)
        )
        
        # Check if maximum episode length is reached
        truncated = self.step_count >= self.max_episode_steps
        self.step_count += 1
        
        info = {
            "angle1": angle1,
            "angle2": angle2,
            "cart_position": (observation[0], observation[1]),
            "angular_velocity": np.linalg.norm(observation[12:18]),
            **reward_info
        }
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info

    def reset_model(self):
        self.step_count = 0
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=noise_low, high=noise_high
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=noise_low, high=noise_high
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()