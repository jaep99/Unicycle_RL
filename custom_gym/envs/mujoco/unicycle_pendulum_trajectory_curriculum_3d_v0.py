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

class UnicyclePendulumTrajectoryCurriculum(MujocoEnv, utils.EzPickle):
    """
    ## Description
    This environment simulates a unicycle with an inverted pendulum attached to it. The goal is to balance the unicycle
    and move it forward to a target distance. The difficulty increases progressively as the agent improves.

    ## Action Space
    The agent takes a 3-element vector for actions:
    | Num | Action                          | Control Min | Control Max | Name (in XML file) | Joint       | Unit         |
    |-----|----------------------------------|-------------|-------------|--------------------|--------------| ------------|
    | 0   | Torque applied on the wheel      | -1          | 1           | wheel_motor        | wheel_joint  | torque (N m) |
    | 1   | Roll stabilization torque        | -1          | 1           | roll_stabilizer    | free_joint   | torque (N m) |
    | 2   | Yaw control torque               | -1          | 1           | yaw_control        | free_joint   | torque (N m) |

    ## Observation Space
    The observation is a 22-element vector:
    | Num | Observation                                        | Min  | Max | Name (in XML file) | Joint | Unit |
    |-----|-----------------------------------------------------|------|-----|-------------------|-------|------|
    | 0   | x-coordinate of the unicycle                       | -Inf | Inf | unicycle | free | position (m) |
    | 1   | y-coordinate of the unicycle                       | -Inf | Inf | unicycle | free | position (m) |
    | 2   | z-coordinate of the unicycle                       | -Inf | Inf | unicycle | free | position (m) |
    | 3   | w-orientation of the unicycle (quaternion)         | -1   | 1   | unicycle | free | unitless |
    | 4   | x-orientation of the unicycle (quaternion)         | -1   | 1   | unicycle | free | unitless |
    | 5   | y-orientation of the unicycle (quaternion)         | -1   | 1   | unicycle | free | unitless |
    | 6   | z-orientation of the unicycle (quaternion)         | -1   | 1   | unicycle | free | unitless |
    | 7   | w-orientation of the pendulum (quaternion)         | -1   | 1   | pendulum | ball | unitless |
    | 8   | x-orientation of the pendulum (quaternion)         | -1   | 1   | pendulum | ball | unitless |
    | 9   | y-orientation of the pendulum (quaternion)         | -1   | 1   | pendulum | ball | unitless |
    | 10  | z-orientation of the pendulum (quaternion)         | -1   | 1   | pendulum | ball | unitless |
    | 11  | angle of the wheel                                 | -Inf | Inf | wheel    | hinge | angle (rad) |
    | 12  | x-velocity of the unicycle                         | -Inf | Inf | unicycle | free | velocity (m/s) |
    | 13  | y-velocity of the unicycle                         | -Inf | Inf | unicycle | free | velocity (m/s) |
    | 14  | z-velocity of the unicycle                         | -Inf | Inf | unicycle | free | velocity (m/s) |
    | 15  | x-angular velocity of the unicycle                 | -Inf | Inf | unicycle | free | angular velocity (rad/s) |
    | 16  | y-angular velocity of the unicycle                 | -Inf | Inf | unicycle | free | angular velocity (rad/s) |
    | 17  | z-angular velocity of the unicycle                 | -Inf | Inf | unicycle | free | angular velocity (rad/s) |
    | 18  | x-angular velocity of the pendulum                 | -Inf | Inf | pendulum | ball | angular velocity (rad/s) |
    | 19  | y-angular velocity of the pendulum                 | -Inf | Inf | pendulum | ball | angular velocity (rad/s) |
    | 20  | z-angular velocity of the pendulum                 | -Inf | Inf | pendulum | ball | angular velocity (rad/s) |
    | 21  | angular velocity of the wheel                      | -Inf | Inf | wheel    | hinge | angular velocity (rad/s) |

    ## Rewards
    The reward function is a weighted sum of the following components:
    1. A balancing reward for keeping the unicycle and pendulum upright
    2. A forward motion reward for moving towards the target distance
    3. A velocity reward for maintaining forward motion
    4. Penalties for excessive tilt and wheel speed

    The weights of these components change as the difficulty increases.

    ## Starting State
    The unicycle starts in a slightly random upright position close to the origin.

    ## Episode Termination
    The episode ends if:
    1. The unicycle or pendulum tilt beyond the current maximum allowed angle
    2. The unicycle reaches the current target distance
    3. The maximum number of steps is reached

    ## Solved Requirements
    The environment is considered solved when the agent can consistently reach the maximum target distance (12m) 
    while maintaining balance.

    """

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 100,
    }

    def __init__(
        self,
        xml_file: str = None,
        frame_skip: int = 2,
        default_camera_config: Dict[str, Union[float, int, np.ndarray]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 0.01,
        **kwargs
    ):
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)

        self.reset_noise_scale = reset_noise_scale

        # Curriculum learning parameters
        self.min_target_distance = 0.1
        self.max_target_distance = 12.0
        self.current_target_distance = self.min_target_distance
        self.min_max_tilt = np.pi / 12  # 15 degrees
        self.max_max_tilt = np.pi / 3   # 60 degrees
        self.current_max_tilt = self.min_max_tilt

        if xml_file is None:
            xml_file = os.path.join(os.path.dirname(__file__), "assets", "unicycle_pendulum_3d.xml")

        observation_space = Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs
        )

        self.metadata = {
            "render_modes": ["human", "rgb_array", "depth_array"],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()

        unicycle_quat = observation[3:7]
        pendulum_quat = observation[7:11]
        
        unicycle_euler = Rotation.from_quat(np.roll(unicycle_quat, -1)).as_euler('xyz')
        pendulum_euler = Rotation.from_quat(np.roll(pendulum_quat, -1)).as_euler('xyz')
        
        unicycle_roll, unicycle_pitch, _ = unicycle_euler
        pendulum_roll, pendulum_pitch, _ = pendulum_euler

        x_position = observation[0]
        x_velocity = observation[12]

        # Compute rewards
        balance_reward = 1.0 - 0.5 * (unicycle_roll**2 + unicycle_pitch**2 + pendulum_roll**2 + pendulum_pitch**2)
        distance_reward = x_position / self.current_target_distance
        velocity_reward = np.clip(x_velocity, 0, 1)  # Reward forward velocity, capped at 1 m/s
        
        # Penalties
        tilt_penalty = -10.0 if (abs(unicycle_roll) > self.current_max_tilt or 
                                 abs(unicycle_pitch) > self.current_max_tilt or
                                 abs(pendulum_roll) > self.current_max_tilt or 
                                 abs(pendulum_pitch) > self.current_max_tilt) else 0.0
        wheel_speed_penalty = -0.1 * observation[21]**2  # Penalty for excessive wheel speed

        # Combine rewards
        reward = balance_reward + distance_reward + velocity_reward + tilt_penalty + wheel_speed_penalty

        # Check termination conditions
        terminated = bool(
            abs(unicycle_roll) > self.current_max_tilt or
            abs(unicycle_pitch) > self.current_max_tilt or
            abs(pendulum_roll) > self.current_max_tilt or
            abs(pendulum_pitch) > self.current_max_tilt or
            x_position >= self.current_target_distance
        )

        info = {
            "x_position": x_position,
            "unicycle_roll": unicycle_roll,
            "unicycle_pitch": unicycle_pitch,
            "pendulum_roll": pendulum_roll,
            "pendulum_pitch": pendulum_pitch,
            "target_distance": self.current_target_distance,
            "max_tilt": self.current_max_tilt,
        }

        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self.reset_noise_scale
        noise_high = self.reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)

        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def update_difficulty(self, success_rate):
        if success_rate > 0.8:
            self.current_target_distance = min(self.current_target_distance + 0.1, self.max_target_distance)
            self.current_max_tilt = min(self.current_max_tilt + np.pi/60, self.max_max_tilt)
        elif success_rate < 0.2:
            self.current_target_distance = max(self.current_target_distance - 0.1, self.min_target_distance)
            self.current_max_tilt = max(self.current_max_tilt - np.pi/60, self.min_max_tilt)

class CurriculumLearning:
    def __init__(self, env, update_interval=100):
        self.env = env
        self.update_interval = update_interval
        self.episode_count = 0
        self.success_count = 0

    def update(self):
        self.episode_count += 1
        if self.episode_count % self.update_interval == 0:
            success_rate = self.success_count / self.update_interval
            self.env.update_difficulty(success_rate)
            self.success_count = 0
            self.episode_count = 0

    def record_success(self):
        self.success_count += 1