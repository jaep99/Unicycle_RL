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

class UnicycleTurningTrajectory(MujocoEnv, utils.EzPickle):
    """
    ## Description
    This environment simulates a unicycle. The goal is to balance the unicycle and move it to y-coordinate 12.

    ## Action Space
    The agent takes a 3-element vector for actions.
    | Num | Action                          | Control Min | Control Max | Name (in XML file) | Joint       | Unit         |
    |-----|----------------------------------|-------------|-------------|--------------------|--------------| ------------|
    | 0   | Torque applied on the wheel      | -1          | 1           | wheel_motor        | wheel_joint  | torque (N m) |
    | 1   | Roll stabilization torque        | -1          | 1           | roll_stabilizer    | free_joint   | torque (N m) |
    | 2   | Yaw control torque               | -1          | 1           | yaw_control        | free_joint   | torque (N m) |

    ## Observation Space
    The observation consists of:
    | Num | Observation                                   | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
    |-----|-----------------------------------------------|------|-----|----------------------------------|-------|---------------------------|
    | 0   | position of the unicycle in the x-direction   | -Inf | Inf | unicycle | free | position (m) |
    | 1   | position of the unicycle in the y-direction   | -Inf | Inf | unicycle | free | position (m) |
    | 2   | position of the unicycle in the z-direction   | -Inf | Inf | unicycle | free | position (m) |
    | 3   | w-component of the unicycle's quaternion      | -1   | 1   | unicycle (quaternion) | free | quaternion component |
    | 4   | x-component of the unicycle's quaternion      | -1   | 1   | unicycle (quaternion) | free | quaternion component |
    | 5   | y-component of the unicycle's quaternion      | -1   | 1   | unicycle (quaternion) | free | quaternion component |
    | 6   | z-component of the unicycle's quaternion      | -1   | 1   | unicycle (quaternion) | free | quaternion component |
    | 7   | angle of the wheel                            | -Inf | Inf | wheel | hinge | angle (rad) |
    | 8   | velocity of the unicycle in the x-direction   | -Inf | Inf | unicycle | free | velocity (m/s) |
    | 9   | velocity of the unicycle in the y-direction   | -Inf | Inf | unicycle | free | velocity (m/s) |
    | 10  | velocity of the unicycle in the z-direction   | -Inf | Inf | unicycle | free | velocity (m/s) |
    | 11  | angular velocity of the unicycle about x-axis | -Inf | Inf | unicycle | free | angular velocity (rad/s) |
    | 12  | angular velocity of the unicycle about y-axis | -Inf | Inf | unicycle | free | angular velocity (rad/s) |
    | 13  | angular velocity of the unicycle about z-axis | -Inf | Inf | unicycle | free | angular velocity (rad/s) |
    | 14  | angular velocity of the wheel                 | -Inf | Inf | wheel | hinge | angular velocity (rad/s) |

    The observation is a `ndarray` with shape `(15,)` where the elements correspond to the table above.

    Note:
    - The unicycle's position and orientation represent its state in 3D space.
    - The wheel's angle and angular velocity represent its rotation around its axis.
    - Quaternions are used to represent 3D orientations to avoid gimbal lock.
    - All angular measurements are in radians.

    ## Rewards
    The reward function encourages balancing the unicycle and moving towards the goal position along the y-axis.

    ## Starting State
    The unicycle starts in a slightly random position and orientation.

    ## Episode Termination
    The episode ends when:
    1. The unicycle falls over (roll or pitch angle exceeds π/3)
    2. The unicycle reaches the goal position (y-coordinate 12)
    3. The maximum number of steps is reached

    ## Solved Requirement
    The environment is considered solved when the agent can keep the unicycle balanced and move it to the goal position along the y-axis.
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
        max_steps: int = 30000,
        **kwargs,
    ):
        if xml_file is None:
            xml_file = os.path.join(os.path.dirname(__file__), "assets", "unicycle_yaw_control_3d.xml")
        
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, max_steps, **kwargs)
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float64)
        self.action_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self._reset_noise_scale = reset_noise_scale
        self.max_steps = max_steps
        self.steps = 0
        self.goal_y = 12.0  # Target 12m along y-axis

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

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.steps += 1
        
        observation = self._get_obs()
        
        y_pos = observation[1]
        y_vel = observation[9]
        quat_mujoco = observation[3:7]
        quat_scipy = np.roll(quat_mujoco, -1)
        euler = Rotation.from_quat(quat_scipy).as_euler('xyz')
        unicycle_roll, unicycle_pitch, _ = euler
        
        # Simplified balancing reward
        balance_reward = 1.0 - 0.5 * (unicycle_roll**2 + unicycle_pitch**2)
        
        # Simplified forward motion reward
        forward_reward = y_vel  # Directly reward forward velocity along y-axis
        
        # Simplified goal reaching reward
        goal_reward = -abs(self.goal_y - y_pos) / self.goal_y  # Normalize by goal distance
        
        # Compute total reward
        reward = balance_reward + forward_reward + goal_reward
        
        # Termination conditions
        terminated = bool(
            abs(unicycle_roll) > np.pi/3 or
            abs(unicycle_pitch) > np.pi/3 or
            abs(y_pos - self.goal_y) < 0.1  # Goal reached
        )
        
        truncated = bool(self.steps >= self.max_steps)
        
        if abs(y_pos - self.goal_y) < 0.1:
            reward += 100  # Bonus for reaching the goal

        info = {
            "y_position": y_pos,
            "y_velocity": y_vel,
            "balance": balance_reward,
            "forward": forward_reward,
            "goal": goal_reward,
            "steps": self.steps,
            "goal_reached": abs(y_pos - self.goal_y) < 0.1
        }
        
        return observation, reward, terminated, truncated, info

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
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()