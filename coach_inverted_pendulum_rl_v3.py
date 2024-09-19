import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Box
from gymnasium import Env
from gymnasium.wrappers import EnvCompatibility
from stable_baselines3 import SAC, TD3, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import os
import argparse
import custom_gym.envs.mujoco
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
import datetime
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import json

"""
from custom_gym.envs.mujoco.coach_inverted_pendulum_3d_v0 import(
    InvertedPendulum3DEnv,
    CoachAgent,
    InvertedPendulum3DEnvWithCoach
)
"""
"""
# Coachagent
class CoachAgent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        # Define a combined space that concatenates observation and action
        low = np.concatenate([observation_space.low, action_space.low])
        high = np.concatenate([observation_space.high, action_space.high])
        combined_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Initialize SAC with the combined space
        self.model = SAC('MlpPolicy', combined_space, verbose=1, device='cpu')
    
    def select_action(self, observation, student_action):
        # Concatenate the observation and student action
        coach_observation = np.concatenate([observation, student_action])
        action, _ = self.model.predict(coach_observation, deterministic=True)
        return action
    
    def train(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps)
"""

class CoachEnvWrapper(gym.Env):
    """
    Custom environment to handle combined observation (student observation + student action)
    for the coach agent.
    """
    def __init__(self, env, action_space):
        self.env = env
        self.action_space = action_space
        
        # Redefine observation space to include both student observation and action
        obs_shape = self.env.observation_space.shape[0]
        action_shape = self.action_space.shape[0]
        combined_obs_shape = obs_shape + action_shape
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(combined_obs_shape,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Reset the environment and return the initial observation
        obs, info = self.env.reset(seed=seed, options=options)
        return np.concatenate([obs, np.zeros(self.action_space.shape)]), info  # Initialize with zero actions

    def step(self, action):
        # Perform a step in the environment, returning the combined observation (obs + student action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        combined_obs = np.concatenate([obs, action])  # Combine observation with student's action
        return combined_obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        return self.env.render(mode=mode)


class CoachAgent:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

        # Wrap the base environment with the custom CoachEnvWrapper
        self.coach_env = CoachEnvWrapper(env, self.action_space)

        # Initialize SAC with the wrapped environment
        self.model = SAC("MlpPolicy", self.coach_env, verbose=1, device="cpu")
        
        # Initialize SAC with the environment instead of spaces
        #self.model = SAC('MlpPolicy', env, verbose=1, device='cpu')
    
    def select_action(self, observation, student_action):
        # Concatenate the observation and student action
        combined_observation = np.concatenate([observation, student_action])
        action, _ = self.model.predict(combined_observation, deterministic=True)
        #action, _ = self.model.predict(coach_observation, deterministic=True)
        return action
    
    def train(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps) # Updates internal policy



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

    def __init__(self, env, coach_agent, logger=None, render_mode=None, **kwargs):
        super(InvertedPendulum3DEnvWithCoach, self).__init__(env)
        self.coach_agent = coach_agent
        self.logger = logger 
        self._render_mode = render_mode
        self.current_episode_reward = 0  # Track the student's current episode reward
        self.previous_episode_reward = None  # Track the student's previous episode reward
        self.coach_rewards =[] # Store coach rewards
        self.coach_total_rewards = 0

    def reset(self, **kwargs):
        # Reset the wrapped environment (InvertedPendulum3DEnv)
        obs, info = self.env.reset(**kwargs)
        
        # Reset the current episode reward
        self.current_episode_reward = 0
        
        self.coach_total_rewards = 0
        self.coach_rewards = []
        
        return obs, info

    def step(self, action):
        """
        state = self.env.state
        coach_action = self.coach_agent.select_action(action, state)
        combined_action = action + coach_action  # Combine student and coach actions
        return self.env.step(combined_action)
        """
        # Use env.unwrapped to access the base environment
        observation = self.env.unwrapped._get_obs()
        
        
        # Get the coach action
        coach_action = self.coach_agent.select_action(observation, action)
        
        # Perform the environment step with the combined action
        combined_action = action + coach_action
        next_observation, student_reward, terminated, truncated, info = self.env.step(combined_action)
        done = terminated or truncated
        
        # Update the student's current episode cumulative reward
        self.current_episode_reward += student_reward

        # Log coach reward
        if self.previous_episode_reward is not None:
            improvement = self.current_episode_reward - self.previous_episode_reward
            coach_reward = improvement
        else:
            coach_reward = 0 # No reward for the first episode

        # Storing coach reward for this step
        self.coach_rewards.append(coach_reward)
        self.coach_total_rewards += coach_reward
        
        # Log coach reward at each step
        print(f"Student reward: {student_reward}, Coach reward: {coach_reward}")
        if self.logger:
            self.logger.record("student/step_reward", student_reward)
            self.logger.record("coach/step_reward", coach_reward)
            self.logger.record("student/timesteps", self.env.unwrapped.step_count)  # Track timestep at each step
            self.logger.record("coach/timesteps", self.env.unwrapped.step_count)    # Track timestep at each step
            if self.env.unwrapped.step_count % 10 == 0:
                self.logger.dump()

        if terminated:
            # Log episode rewards
            if self.logger:
                # Student's rewards
                self.logger.record("student/episode_reward", self.current_episode_reward)
                self.logger.record("student/ep_rew_mean", self.current_episode_reward / self.env.unwrapped.step_count)
                
                # Coach's rewards
                self.logger.record("coach/episode_reward", self.coach_total_rewards)
                self.logger.record("coach/ep_rew_mean", self.coach_total_rewards / self.env.unwrapped.step_count)
                
                if self.env.unwrapped.step_count % 10 == 0:
                    self.logger.dump()

            # Create combined observation for replay buffer (observation + action)
            combined_observation = np.concatenate([observation, action])
            combined_next_observation = np.concatenate([next_observation, action])

            # Log the coach reward and train the coach
            self.coach_agent.model.replay_buffer.add(
                combined_observation, # Comb observation
                combined_next_observation, # Next observation
                combined_action, # Combined action
                np.array([coach_reward]), # Coach reward
                np.array([done]), # Episode termination status
                [info] # Additional info
            )

            # Update the previous episode reward
            self.previous_episode_reward = self.current_episode_reward
            self.current_episode_reward = 0  # Reset for the next episode
            self.coach_total_rewards = 0
        
        return next_observation, student_reward, terminated, False, info
    
    def render(self):
        return self.env.render()
        """
        if self._render_mode:
            return self.env.render(mode=self._render_mode)
        """

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Max episode steps added
MAX_EPISODE_STEPS = 30000


def train(env, sb3_algo):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{sb3_algo}_{timestamp}"

    # Define observation and action spaces for the coach
    observation_space = env.observation_space
    action_space = env.action_space

    # Create a coach agent
    coach_agent = CoachAgent(env)

    # TensorBoard logging for custom rewards
    log_dir = f"logs/{run_name}_coach"
    logger = configure(log_dir, ["tensorboard"])

    # Wrap the environment with the coach
    wrapped_env = InvertedPendulum3DEnvWithCoach(env, coach_agent, logger)

    # Create seperated environment for model evaluation added
    #eval_env = gym.make('InvertedPendulum3D-v1', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)
    eval_env = InvertedPendulum3DEnvWithCoach(
        gym.make('InvertedPendulum3D-v3', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS),
        coach_agent,
        logger=None # No logging during evaluation
    )

    eval_env = Monitor(eval_env)

    # EvalCallback added
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"{model_dir}/best_{run_name}",
                                 log_path=log_dir, eval_freq=10000,
                                 deterministic=True, render=False)

    match sb3_algo:
        case 'SAC':
            model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'TD3':
            model = TD3('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'A2C':
            model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case _:
            print('Algorithm not found')
            return

    TIMESTEPS = 100000 
    iters = 0

    while True:
        iters += 1
        
        # Train the student model
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=run_name, callback=eval_callback)
        
        # Train the coach
        coach_agent.train(total_timesteps=TIMESTEPS)
        
        model.save(f"{model_dir}/{run_name}_{TIMESTEPS*iters}")

        # Log coach rewards to TensorBoard
        if wrapped_env.coach_rewards:  
            coach_reward = wrapped_env.coach_rewards[-1]
            if logger:
                logger.record("coach/episode_reward", coach_reward)

        # Log student rewards to TensorBoard
        if wrapped_env.current_episode_reward is not None: 
            student_reward = wrapped_env.current_episode_reward
            if logger:
                logger.record("student/episode_reward", student_reward)

        if logger:
            logger.dump(step=iters * TIMESTEPS)
            #logger.dump(step=self.env.unwrapped.step_count)
    
    logger.close()
    
   


def test(env, sb3_algo, path_to_model):
    observation_space = env.observation_space
    action_space = env.action_space
    coach_agent = CoachAgent(env)
    gymenv = InvertedPendulum3DEnvWithCoach(env, coach_agent, logger=None)

    match sb3_algo:
        case 'SAC':
            model = SAC.load(path_to_model, env=gymenv)
        case 'TD3':
            model = TD3.load(path_to_model, env=gymenv)
        case 'A2C':
            model = A2C.load(path_to_model, env=gymenv)
        case _:
            print('Algorithm not found')
            return

    obs, _ = gymenv.reset()
    terminated = truncated = False
    total_reward = 0
    step_count = 0
    
    while not (terminated or truncated) and step_count < MAX_EPISODE_STEPS:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = gymenv.step(action)
        gymenv.render()
        
        total_reward += reward
        step_count += 1
        
        cart_x, cart_y = obs[0], obs[1]

        # Print step information
        print(f"Step: {step_count}, Reward: {reward:.4f}, Angle: {info.get('angle', 'N/A'):.4f}, Position: ({cart_x:.4f}, {cart_y:.4f})")
        
    print(f"\nEpisode finished after {step_count} steps")
    print(f"Total reward: {total_reward}")
    print(f"Average reward per step: {total_reward / step_count}")

if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    #gymenv = gym.make('InvertedPendulum3DWithCoach-v1', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)
    #gymenv = gym.make('InvertedPendulum3DWithCoach-v1', render_mode=None)
    base_env = gym.make('InvertedPendulum3D-v3', render_mode=None)

    #base_env = EnvCompatibility(base_env)
    check_env(base_env)
    
    

    # Wrap the environment with the coach
    #gymenv = InvertedPendulum3DEnvWithCoach(base_env, coach_agent, render_mode=None)
    #gymenv = InvertedPendulum3DEnvWithCoach(base_env, coach_agent)

    if args.train:
        # TensorBoard logging for training
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.sb3_algo}_{timestamp}"
        log_dir = f"logs/{run_name}_coach"
        logger = configure(log_dir, ["tensorboard"])

        # Create a coach agent
        coach_agent = CoachAgent(base_env)

        # Wrap the environment with the coach and logger
        gymenv = InvertedPendulum3DEnvWithCoach(base_env, coach_agent, logger)

        train(gymenv, args.sb3_algo)

    if args.test:
        if os.path.isfile(args.test):
            #gymenv = gym.make('InvertedPendulum3DWithCoach-v1', render_mode='human', max_episode_steps=MAX_EPISODE_STEPS)
            #gymenv = gym.make('InvertedPendulum3DWithCoach-v1', render_mode='human')
            test_env = gym.make('InvertedPendulum3D-v3', render_mode='human')
            #gymenv = InvertedPendulum3DEnvWithCoach(base_env, coach_agent, render_mode='human')
            # Unwrap the environment if it's wrapped with any wrapper like TimeLimit, OrderEnforcing, or PassiveEnvChecker
            coach_agent = CoachAgent(base_env)
            gymenv = InvertedPendulum3DEnvWithCoach(base_env, coach_agent, logger=None)
            test(gymenv, args.sb3_algo, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')