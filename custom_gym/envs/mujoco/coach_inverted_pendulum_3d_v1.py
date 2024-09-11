import os
from typing import Dict, Union

import numpy as np
from scipy.spatial.transform import Rotation

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 0.6)),
}

class InvertedPendulum3DWithCoach(MujocoEnv, utils.EzPickle):
    """
    ## Description
    This environment is a 3D extension of the classic Inverted Pendulum environment with an additional coach agent.
    The goal is for the student agent to learn to balance a pole on a cart that can move linearly in the x-y plane,
    with assistance from a coach agent.

    ## Action Space
    Both the student and coach agents take a 2-element vector for actions.
    The action space for each agent is a continuous `(action_x, action_y)` in `[-9.9, 9.9]^2`, where `action_x` and `action_y`
    represent the numerical force applied to the cart in the x and y directions respectively.

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit) |
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-------------|
    | 0   | Force applied on the cart in x-direction | -2 | 2  | slider_x | slide | Force (N) |
    | 1   | Force applied on the cart in y-direction | -2 | 2  | slider_y | slide | Force (N) |

    ## Coach Interaction
    The coach agent provides physical assistance by applying additional force to the cart.
    The final action applied to the environment is a weighted combination of the student's action and the coach's action.
    The coach's influence can be adjusted and potentially decreased over time as the student improves.

    ## Observation Space
    The observation space remains unchanged and consists of positional and velocity values of the cart and pole.
    The pole's orientation is represented using a quaternion.
    | Num | Observation                                   | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
    | --- | --------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------- |
    | 0   | position of the cart in the x-direction       | -Inf | Inf | slider_x | slide | position (m) |
    | 1   | position of the cart in the y-direction       | -Inf | Inf | slider_y | slide | position (m) |
    | 2   | w-component of the pole's quaternion          | -Inf | Inf | ball (quaternion) | ball | quaternion component |
    | 3   | x-component of the pole's quaternion          | -Inf | Inf | ball (quaternion) | ball | quaternion component |
    | 4   | y-component of the pole's quaternion          | -Inf | Inf | ball (quaternion) | ball | quaternion component |
    | 5   | z-component of the pole's quaternion          | -Inf | Inf | ball (quaternion) | ball | quaternion component |
    | 6   | velocity of the cart in the x-direction       | -Inf | Inf | slider_x | slide | velocity (m/s) |
    | 7   | velocity of the cart in the y-direction       | -Inf | Inf | slider_y | slide | velocity (m/s) |
    | 8   | angular velocity of the pole about x-axis     | -Inf | Inf | ball | ball | angular velocity (rad/s) |
    | 9   | angular velocity of the pole about y-axis     | -Inf | Inf | ball | ball | angular velocity (rad/s) |
    | 10  | angular velocity of the pole about z-axis     | -Inf | Inf | ball | ball | angular velocity (rad/s) |

    [observation space table remains the same]

    ## Rewards
    The reward structure remains unchanged. 

    ## Starting State
    The initial position and velocity of the cart and pole are randomly sampled from a Gaussian with zero mean and a standard deviation of `reset_noise_scale`.

    ## Episode End
    The episode ends if any of the following occurs:
    1. Any of the state space values are no longer finite.
    2. The absolute value of the vertical angle between the pole and the cart is greater than 0.3 radians.

    ## Learning Objective
    The ultimate goal is for the student agent to learn to balance the pole without assistance from the coach.
    This is achieved by gradually reducing the coach's influence over the course of training.
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
            xml_file = os.path.join(os.path.dirname(__file__), "assets", "inverted_pendulum_3d.xml")

        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)

        self._reset_noise_scale = reset_noise_scale
        self.coach_model = None  # Using model instead of action to provide an actual policy with an integrated parameter.

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

    def compute_reward(self, observation, action, angle):
        angle_reward = np.cos(angle)
        
        cart_x, cart_y = observation[0], observation[1]
        distance_from_center = np.sqrt(cart_x**2 + cart_y**2)
        position_penalty = -5.0 * distance_from_center**2  
        
        angular_velocity = np.linalg.norm(observation[8:11])
        angular_velocity_penalty = -0.1 * angular_velocity
        
        action_penalty = -0.01 * np.sum(np.square(action))
        
        center_bonus = 1.0 if distance_from_center < 0.1 else 0.0 
        
        reward = angle_reward + position_penalty + angular_velocity_penalty + action_penalty + center_bonus
        
        return reward

    def step(self, action):
        if self.coach_model is not None:
            obs = self._get_obs()
            coach_action, _ = self.coach_model.predict(obs, deterministic=True)
            combined_action = action + coach_action
            self.coach_action = coach_action 
        else:
            combined_action = action
            self.coach_action = np.zeros_like(action)

        self.do_simulation(combined_action, self.frame_skip)
        
        observation = self._get_obs()
        
        quat = observation[2:6]
        quat_xyzw = np.roll(quat, -1)
        r = Rotation.from_quat(quat_xyzw)
        angle = r.magnitude()
        
        reward = self.compute_reward(observation, combined_action, angle)
        
        terminated = bool(
            not np.isfinite(observation).all() or 
            (angle > 0.4) or 
            (np.abs(observation[0]) > 5) or 
            (np.abs(observation[1]) > 5)
        )
        
        info = {"angle": angle, "coach_action": self.coach_action}
        
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
        self.coach_action = None  # Coach part
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()

    def set_coach_model(self, model):   # Coach model method
        self.coach_model = model