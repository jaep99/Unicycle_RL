import os
from typing import Dict, Union

import numpy as np
from scipy.spatial.transform import Rotation

import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 0.6)),
}


class InvertedPendulum3DEnv(MujocoEnv, utils.EzPickle):
    """
    ## Description
    This environment is a 3D extension of the classic Inverted Pendulum environment.
    The goal is to balance a pole on a cart that can move linearly in the x-y plane.

    ## V0 version mainly focus on the time factor for locating the object in the center.

    ## Action Space
    The agent takes a 2-element vector for actions.
    The action space is a continuous `(action_x, action_y)` in `[-5, 5]^2`, where `action_x` and `action_y`
    represent the numerical force applied to the cart in the x and y directions respectively.

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit) |
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-------------|
    | 0   | Force applied on the cart in x-direction | -5 | 5  | slider_x | slide | Force (N) |
    | 1   | Force applied on the cart in y-direction | -5 | 5  | slider_y | slide | Force (N) |

    ## Observation Space
    The observation space consists of positional and velocity values of the cart and pole.
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

    ## Rewards
    A reward of +1 is given for each timestep where the pole is upright.
    The pole is considered upright if the Euclidean distance between its current quaternion and the upright quaternion (0, 0, 0, 1) is less than 0.2.

    ## Starting State
    The initial position and velocity of the cart and pole are randomly sampled from a Gaussian with zero mean and a standard deviation of `reset_noise_scale`.

    ## Episode End
    The episode ends if any of the following occurs:
    1. Any of the state space values are no longer finite.
    2. The absolute value of the vertical angle between the pole and the cart is greater than 0.3 radians.

    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        #"render_fps": 50,
        
    }

    def __init__(
        self,
        xml_file: str = None,
        frame_skip: int = 2,
        default_camera_config: Dict[str, Union[float, int, np.ndarray]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 0.02, # Noise scale increased from 0.01 -> 0.02
        **kwargs,
    ):
        if xml_file is None:
            xml_file = os.path.join(os.path.dirname(__file__), "assets", "inverted_pendulum_3d.xml")

        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)
        
        # 5 DOF
        observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)

        self._reset_noise_scale = reset_noise_scale
        self.step_count = 0  # Initialize step_count here
        self.max_steps = 3000  # Add this line to define max_steps
        #self.dt = 0.02

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        """
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        """
        # Calculate dt based on the initialized timestep and frame_skip
        computed_dt = self.model.opt.timestep * frame_skip
        print('Computed_dt', computed_dt)

        # Update render_fps to match dt
        self.metadata["render_fps"] = int(np.round(1.0 / computed_dt))
        print('render_fps', self.metadata["render_fps"])
        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }

    # Reward algorithms
    def compute_reward(self, observation, action, angle):
        angle_reward = np.cos(angle)
        
        cart_x, cart_y = observation[0], observation[1]
        time_factor = min(10.0, self.step_count / self.max_steps)
        position_penalty = -time_factor * (cart_x**2 + cart_y**2)
        
        angular_velocity = np.linalg.norm(observation[8:11])
        angular_velocity_penalty = -0.1 * angular_velocity
        
        action_penalty = -0.01 * np.sum(np.square(action))
        
        survival_bonus = 0.1
        
        reward = angle_reward + position_penalty + angular_velocity_penalty + action_penalty + survival_bonus
        
        return reward

    def step(self, action, coach_action=None):
        if coach_action is not None:
            action += coach_action
            
        self.do_simulation(action, self.frame_skip)
        self.step_count += 1

        observation = self._get_obs()
        
        # Obtaining quaternion values from the observation (w, x, y, z)
        quat = observation[2:6]
        
        # Transformation for "scipy" (x, y, z, w)
        quat_xyzw = np.roll(quat, -1)
        
        # Quaternion to Euler
        r = Rotation.from_quat(quat_xyzw)
        
        # Radian Conversion
        angle = r.magnitude()
        
        # Reward Calculation
        reward = self.compute_reward(observation, action, angle)
        
        # Terminate if angle > 0.4 radian or goes beyond spaces
        terminated = bool(
            not np.isfinite(observation).all() or 
            (angle > 0.4) or 
            (np.abs(observation[0]) > 5) or 
            (np.abs(observation[1]) > 5)
        )
        
        info = {"angle": angle}
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, False, info

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
    

# Coachagent
class CoachAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def select_action(self, student_action, angle):
        # Coach policy: Determine when and how much to intervene
        # Example heuristic: apply additional torque if angle exceeds threshold
        angle_threshold = 0.1
        if np.abs(angle) > angle_threshold:  # Assuming state[3] is the angle
            assist_action = self.action_space.sample() * 0.1  # Small corrective torque
        else:
            assist_action = np.zeros_like(student_action)
        return assist_action

# Wrapper for the environment to include the CoachAgent
class InvertedPendulum3DEnvWithCoach(gym.Wrapper):
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        #"render_fps": 60,
    }
    """

    def __init__(self, env, coach_agent, render_mode=None, **kwargs):
        super(InvertedPendulum3DEnvWithCoach, self).__init__(env)
        self.coach_agent = coach_agent
        #self._render_mode = render_mode

    def step(self, action):
        """
        state = self.env.state
        coach_action = self.coach_agent.select_action(action, state)
        combined_action = action + coach_action  # Combine student and coach actions
        return self.env.step(combined_action)
        """
        # Use env.unwrapped to access the base environment
        observation = self.env.unwrapped._get_obs()
        
        # Obtain the quaternion values
        quat = observation[2:6]
        quat_xyzw = np.roll(quat, -1)  # Convert to (x, y, z, w)
        r = Rotation.from_quat(quat_xyzw)
        angle = r.magnitude()  # Calculate angle using the Rotation object
        
        # Use angle
        coach_action = self.coach_agent.select_action(action, angle)
        
        # Perform the environment step with the combined action
        combined_action = action + coach_action
        return self.env.step(combined_action)
    
    def render(self):
        return self.env.render()
        """
        if self._render_mode:
            return self.env.render(mode=self._render_mode)
        """