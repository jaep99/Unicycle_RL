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

class UnicyclePendulumBalance(MujocoEnv, utils.EzPickle):
    """
    ## Description
    This environment simulates a unicycle with an inverted pendulum attached to it. The goal is to balance both the unicycle and the pendulum while controlling the unicycle's movement.

    ## Action Space
    The agent takes a 3-element vector for actions.
    | Num | Action                          | Control Min | Control Max | Name (in XML file) | Joint       | Unit         |
    |-----|----------------------------------|-------------|-------------|--------------------|--------------| ------------|
    | 0   | Torque applied on the wheel      | -1          | 1           | wheel_motor        | wheel_joint  | torque (N m) |
    | 1   | Roll stabilization torque        | -1          | 1           | roll_stabilizer    | free_joint   | torque (N m) |
    | 2   | Yaw control torque               | -1          | 1           | yaw_control        | free_joint   | torque (N m) |

    ## Observation Space
    The observation consists of:
    | Num | Observation                                        | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
    |-----|-----------------------------------------------------|------|-----|----------------------------------|-------|---------------------------|
    | 0   | position of the unicycle in the x-direction         | -Inf | Inf | unicycle | free | position (m) |
    | 1   | position of the unicycle in the y-direction         | -Inf | Inf | unicycle | free | position (m) |
    | 2   | position of the unicycle in the z-direction         | -Inf | Inf | unicycle | free | position (m) |
    | 3   | w-component of the unicycle's quaternion            | -1   | 1   | unicycle (quaternion) | free | quaternion component |
    | 4   | x-component of the unicycle's quaternion            | -1   | 1   | unicycle (quaternion) | free | quaternion component |
    | 5   | y-component of the unicycle's quaternion            | -1   | 1   | unicycle (quaternion) | free | quaternion component |
    | 6   | z-component of the unicycle's quaternion            | -1   | 1   | unicycle (quaternion) | free | quaternion component |
    | 7   | w-component of the pendulum's quaternion            | -1   | 1   | pendulum (quaternion) | ball | quaternion component |
    | 8   | x-component of the pendulum's quaternion            | -1   | 1   | pendulum (quaternion) | ball | quaternion component |
    | 9   | y-component of the pendulum's quaternion            | -1   | 1   | pendulum (quaternion) | ball | quaternion component |
    | 10  | z-component of the pendulum's quaternion            | -1   | 1   | pendulum (quaternion) | ball | quaternion component |
    | 11  | angle of the wheel                                  | -Inf | Inf | wheel | hinge | angle (rad) |
    | 12  | velocity of the unicycle in the x-direction         | -Inf | Inf | unicycle | free | velocity (m/s) |
    | 13  | velocity of the unicycle in the y-direction         | -Inf | Inf | unicycle | free | velocity (m/s) |
    | 14  | velocity of the unicycle in the z-direction         | -Inf | Inf | unicycle | free | velocity (m/s) |
    | 15  | angular velocity of the unicycle about x-axis       | -Inf | Inf | unicycle | free | angular velocity (rad/s) |
    | 16  | angular velocity of the unicycle about y-axis       | -Inf | Inf | unicycle | free | angular velocity (rad/s) |
    | 17  | angular velocity of the unicycle about z-axis       | -Inf | Inf | unicycle | free | angular velocity (rad/s) |
    | 18  | angular velocity of the pendulum about x-axis       | -Inf | Inf | pendulum | ball | angular velocity (rad/s) |
    | 19  | angular velocity of the pendulum about y-axis       | -Inf | Inf | pendulum | ball | angular velocity (rad/s) |
    | 20  | angular velocity of the pendulum about z-axis       | -Inf | Inf | pendulum | ball | angular velocity (rad/s) |
    | 21  | angular velocity of the wheel                       | -Inf | Inf | wheel | hinge | angular velocity (rad/s) |

    The observation is a `ndarray` with shape `(22,)` where the elements correspond to the table above.

    Note:
    - The unicycle's position and orientation represent its state in 3D space.
    - The pendulum's orientation is represented by a quaternion due to its ball joint.
    - The wheel's angle and angular velocity represent its rotation around its axis.
    - Quaternions are used to represent 3D orientations to avoid gimbal lock.
    - All angular measurements are in radians.

    ## Rewards
    The reward function should be designed to encourage balancing the unicycle and controlling its movement.

    ## Starting State
    The unicycle starts in a slightly random position and orientation.

    ## Episode Termination
    The episode ends when:
    1. The unicycle falls over
    2. The pendulum falls over
    2. The maximum number of steps is reached

    ## Solved Requirement
    The environment is considered solved when the agent can keep the unicycle balanced and control its movement for a certain number of steps.
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
            xml_file = os.path.join(os.path.dirname(__file__), "assets", "unicycle_pendulum_3d.xml")
        
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, max_steps, **kwargs)
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float64)
        self.action_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self._reset_noise_scale = reset_noise_scale
        self.max_steps = max_steps
        self.steps = 0

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
        
        # Extract quaternions and convert to euler angles
        unicycle_quat = observation[3:7]  # Unicycle quaternion (wxyz)
        pendulum_quat = observation[7:11]  # Pendulum quaternion (wxyz)
        
        unicycle_euler = Rotation.from_quat(np.roll(unicycle_quat, -1)).as_euler('xyz')
        pendulum_euler = Rotation.from_quat(np.roll(pendulum_quat, -1)).as_euler('xyz')
        
        unicycle_roll, unicycle_pitch, _ = unicycle_euler
        pendulum_roll, pendulum_pitch, _ = pendulum_euler
        
        # Compute balancing reward for unicycle
        unicycle_balance_reward = 1.0 - 0.5 * (unicycle_roll**2 + unicycle_pitch**2)
        
        # Compute balancing reward for pendulum
        pendulum_balance_reward = 1.0 - 0.5 * (pendulum_roll**2 + pendulum_pitch**2)
        
        # Penalize excessive tilt
        unicycle_tilt_penalty = -10.0 if abs(unicycle_roll) > np.pi/4 or abs(unicycle_pitch) > np.pi/4 else 0.0
        pendulum_tilt_penalty = -10.0 if abs(pendulum_roll) > np.pi/4 or abs(pendulum_pitch) > np.pi/4 else 0.0
        
        # Penalize excessive wheel speed
        wheel_speed_penalty = -0.1 * observation[21]**2  # Last element is wheel angular velocity
        
        # Compute total reward
        reward = unicycle_balance_reward + pendulum_balance_reward + unicycle_tilt_penalty + pendulum_tilt_penalty + wheel_speed_penalty
        
        # Check termination conditions
        terminated = bool(
            abs(unicycle_roll) > np.pi/3 or
            abs(unicycle_pitch) > np.pi/3 or
            abs(pendulum_roll) > np.pi/3 or
            abs(pendulum_pitch) > np.pi/3
        )
        
        truncated = bool(self.steps >= self.max_steps)
        
        if truncated and not terminated:
            reward += 100  # Bonus for staying upright for max_steps
        
        info = {
            "unicycle_roll": unicycle_roll,
            "unicycle_pitch": unicycle_pitch,
            "pendulum_roll": pendulum_roll,
            "pendulum_pitch": pendulum_pitch,
            "unicycle_balance_reward": unicycle_balance_reward,
            "pendulum_balance_reward": pendulum_balance_reward,
            "unicycle_tilt_penalty": unicycle_tilt_penalty,
            "pendulum_tilt_penalty": pendulum_tilt_penalty,
            "wheel_speed_penalty": wheel_speed_penalty,
            "steps": self.steps,
        }
        
        if self.render_mode == "human":
            self.render()
        
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