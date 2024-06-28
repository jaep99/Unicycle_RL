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
    The action space is a continuous `(action_x, action_y)` in `[-3, 3]^2`, where `action_x` and `action_y`
    represent the numerical force applied to the cart in the x and y directions respectively.

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit) |
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-------------|
    | 0   | Force applied on the cart in x-direction | -3 | 3  | slider_x | slide | Force (N) |
    | 1   | Force applied on the cart in y-direction | -3 | 3  | slider_y | slide | Force (N) |

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
    The reward is computed as follows:
    1. A constant survival bonus
    2. A component based on the cosine of the angles of both poles
    3. A penalty based on the cart's distance from the origin
    4. A penalty based on the angular velocities of both poles
    5. A small penalty based on the action to discourage excessive movement

    The exact formula is:
    reward = alive_bonus + cos(angle1) + cos(angle2) - 0.1 * (cart_x^2 + cart_y^2) - 0.1 * |angular_velocity| - 0.01 * |action|^2

    Where alive_bonus is set to 10.0.

    ## Starting State
    The initial position and velocity of the cart and both poles are randomly sampled from a uniform distribution
    with range [-reset_noise_scale, reset_noise_scale].

    ## Episode End
    The episode ends if any of the following occurs:
    1. Any of the state space values are no longer finite.
    2. The absolute value of the angle of the lower pole (angle1) with respect to vertical exceeds 0.4 radians.
    3. The absolute value of the angle of the upper pole (angle2) with respect to vertical exceeds 0.6 radians.
    4. The cart moves more than 10 units away from the origin in either the x or y direction.

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
        **kwargs,
    ):
        if xml_file is None:
            xml_file = os.path.join(os.path.dirname(__file__), "assets", "inverted_double_pendulum_3d.xml")

        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)
        
        # 8 DOF
        observation_space = Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64)

        self._reset_noise_scale = reset_noise_scale

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

        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }

    def compute_reward(self, observation, action, angle1, angle2):
        # Fixed survival bonus
        survival_bonus = 5.0  
        
        angle_reward = np.cos(angle1) + np.cos(angle2)
        
        cart_x, cart_y = observation[0], observation[1]
        position_penalty = 0.1 * (cart_x**2 + cart_y**2)
        
        angular_velocity = np.linalg.norm(observation[12:18])
        vel_penalty = 0.1 * angular_velocity
        
        action_penalty = 0.01 * np.sum(np.square(action))
        
        # Total reward calculation
        reward = survival_bonus + angle_reward - position_penalty - vel_penalty - action_penalty
        
        return reward, {
            "survival_bonus": survival_bonus,
            "angle_reward": angle_reward,
            "posi_penalty": position_penalty,
            "vel_penalty": vel_penalty,
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
        
        info = {
            "angle1": angle1,
            "angle2": angle2,
            "cart_position": (observation[0], observation[1]),
            "angular_velocity": np.linalg.norm(observation[12:18]),
            **reward_info
        }
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, False, info

    def reset_model(self):
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