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

class PendulumTrajectory(MujocoEnv, utils.EzPickle):
    """
    ## Description
    This environment is a modified version of the 3D Inverted Double Pendulum environment.
    The goal is to move a cart with two balanced poles beyond line (x = 12) while keeping the poles upright
    and staying as close as possible to the x-axis (y = 0).

    ## Action Space
    The agent takes a 2-element vector for actions.
    The action space is a continuous `(action_x, action_y)` in `[-15, 15]^2`, where `action_x` and `action_y`
    represent the numerical force applied to the cart in the x and y directions respectively.

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit) |
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-------------|
    | 0   | Force applied on the cart in x-direction | -2 | 2  | slider_x | slide | Force (N) |
    | 1   | Force applied on the cart in y-direction | -2 | 2  | slider_y | slide | Force (N) |

    ## Observation Space
    The observation space remains the same as the original environment, consisting of positional and velocity values
    of the cart and both poles. The poles' orientations are represented using quaternions.

    | Num | Observation                                   | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
    | --- | --------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------- |
    | 0   | position of the cart in the x-direction       | -Inf | Inf | slider_x | slide | position (m) |
    | 1   | position of the cart in the y-direction       | -Inf | Inf | slider_y | slide | position (m) |
    | 2-5 | quaternion components of the first pole       | -Inf | Inf | ball1 (quaternion) | ball | quaternion component |
    | 6-9 | quaternion components of the second pole      | -Inf | Inf | ball2 (quaternion) | ball | quaternion component |
    | 10  | velocity of the cart in the x-direction       | -Inf | Inf | slider_x | slide | velocity (m/s) |
    | 11  | velocity of the cart in the y-direction       | -Inf | Inf | slider_y | slide | velocity (m/s) |
    | 12-14 | angular velocity of the first pole          | -Inf | Inf | ball1 | ball | angular velocity (rad/s) |
    | 15-17 | angular velocity of the second pole         | -Inf | Inf | ball2 | ball | angular velocity (rad/s) |

    ## Rewards
    The reward is computed as follows:
    1. Progress reward: Encourages movement towards x=12
    2. Deviation penalty: Penalizes deviation from y=0
    3. Goal bonus: Large bonus for reaching x≥12
    4. Balance reward: Encourages keeping the poles upright
    5. Velocity control: Penalizes excessive speed

    The exact formula is:
    reward = w1*r_progress + w2*r_deviation + w3*r_goal + w4*r_balance + w5*r_velocity
    Where w1, w2, w3, and w4 are adjustable weights.

    ## Starting State
    The initial position and velocity of the cart and both poles are randomly sampled from a uniform distribution
    with range [-reset_noise_scale, reset_noise_scale].

    ## Episode End
    The episode ends if any of the following occurs:
    1. Any of the state space values are no longer finite.
    2. The cart reaches or exceeds x=12 (successful termination).

    ## Truncation
    The episode is truncated when the cart reaches or exceeds x=12, indicating successful completion of the task.

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
            xml_file = os.path.join(os.path.dirname(__file__), "assets", "pendulum_trajectory_3d.xml")

        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)
        
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

    def compute_reward(self, observation, action, angle1, angle2):
        x, y = observation[0], observation[1]
        vx, vy = observation[10], observation[11]
        
        # Progress reward
        r_progress = 100 * (2 / (1 + np.exp(-0.3 * x)) - 1)
        
        # Deviation penalty
        r_deviation = -50 * abs(y)
        
        # Goal bonus
        r_goal = 100 if x >= 12 else 0

        # Angle reward
        r_angle = 2.0 * (np.cos(angle1)**2 + np.cos(angle2)**2 - 1)
        
        # Velocity control penalty
        r_velocity = -0.1 * (vx**2 + vy**2)
        
        # Combine rewards
        w1, w2, w3, w4, w5 = 1.0, 1.0, 1.0, 1.0, 1.0  # Adjust these weights as needed
        reward = (w1 * r_progress) + (w2 * r_deviation) + (w3 * r_goal) + (w4 * r_angle) + (w5 * r_velocity)
        
        reward_info = {
            "progress_reward": r_progress,
            "deviation_penalty": r_deviation,
            "goal_bonus": r_goal,
            "angle_reward": r_angle,
            "velocity_penalty": r_velocity
        }
        
        return reward, reward_info

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        
        observation = self._get_obs()
        
        # Obtaining quaternion values from the observation for both poles
        quat1 = observation[2:6]
        quat2 = observation[6:10]
        
        # Transformation for scipy format (x, y, z, w)
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
        
        # Terminate if angle1 > 0.6 or angle2 > 0.6 or cart goes beyond spaces
        terminated = bool(
            not np.isfinite(observation).all() or 
            (angle1 > 0.4) or (angle2 > 0.4)
        )
        
        x, y = observation[0], observation[1]

        truncated = bool(x >= 12)
        
        info = {
            **reward_info,
            
            "terminated": terminated,
            "truncated": truncated,
            "termination_reason": "angle_limit" if (angle1 > 0.6 or angle2 > 0.6) else "out_of_bounds" if not np.isfinite(observation).all() else None,
            
            "cart_x": x,
            "cart_y": y,
            "cart_velocity_x": observation[10],
            "cart_velocity_y": observation[11],
            
            "pole1_angle": angle1,
            "pole2_angle": angle2,
            
            "goal_reached": truncated
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