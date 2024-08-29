import gym
from gym import spaces
from gym.spaces import Dict, Box
from stable_baselines3 import SAC, TD3, A2C
import os
import argparse
import custom_gym.envs.mujoco
from stable_baselines3.common.callbacks import EvalCallback
import datetime
import numpy as np

"""
from custom_gym.envs.mujoco.coach_inverted_pendulum_3d_v0 import(
    InvertedPendulum3DEnv,
    CoachAgent,
    InvertedPendulum3DEnvWithCoach
)
"""

# Coachagent
class CoachAgent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        # Define a combined space that concatenates observation and action
        low = np.concatenate([observation_space.low, action_space.low])
        high = np.concatenate([observation_space.high, action_space.high])
        combined_space = Box(low=low, high=high, dtype=np.float32)

        # Initialize SAC with the combined space
        self.model = SAC('MlpPolicy', combined_space, verbose=1, device='cpu')
    
    def select_action(self, observation, student_action):
        # Concatenate the observation and student action
        coach_observation = np.concatenate([observation, student_action])
        action, _ = self.model.predict(coach_observation, deterministic=True)
        return action
    
    def train(self, total_timesteps=10000):
        self.model.learn(total_timesteps=total_timesteps)


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
        self._render_mode = render_mode
        self.current_episode_reward = 0  # Track the student's current episode reward
        self.previous_episode_reward = None  # Track the student's previous episode reward

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
        observation, student_reward, terminated, truncated, info = self.env.step(combined_action)
        
        # Update the student's current episode cumulative reward
        self.current_episode_reward += student_reward
        
        if terminated or truncated:
            # Calculate coach's reward based on student's improvement
            if self.previous_episode_reward is not None:
                improvement = self.current_episode_reward - self.previous_episode_reward
                coach_reward = improvement
            else:
                coach_reward = 0  # No reward for the first episode

            # Log the coach reward and train the coach
            self.coach_agent.model.replay_buffer.add(np.concatenate([observation, combined_action]), coach_reward)
            
            # Update the previous episode reward
            self.previous_episode_reward = self.current_episode_reward
            self.current_episode_reward = 0  # Reset for the next episode
        
        return observation, student_reward, terminated, truncated, info
    
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
    observation_space = base_env.observation_space
    action_space = base_env.action_space

    # Create a coach agent
    coach_agent = CoachAgent(observation_space, action_space)

    # Wrap the environment with the coach
    wrapped_env = InvertedPendulum3DEnvWithCoach(env, coach_agent)

    # Create seperated environment for model evaluation added
    #eval_env = gym.make('InvertedPendulum3D-v1', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)
    eval_env = InvertedPendulum3DEnvWithCoach(
        gym.make('InvertedPendulum3D-v3', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS),
        coach_agent
    )

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

def test(env, sb3_algo, path_to_model):
    observation_space = env.observation_space
    action_space = env.action_space
    coach_agent = CoachAgent(observation_space, action_space)
    wrapped_env = InvertedPendulum3DEnvWithCoach(env, coach_agent)

    match sb3_algo:
        case 'SAC':
            model = SAC.load(path_to_model, env=env)
        case 'TD3':
            model = TD3.load(path_to_model, env=env)
        case 'A2C':
            model = A2C.load(path_to_model, env=env)
        case _:
            print('Algorithm not found')
            return

    obs, _ = wrapped_env.reset()
    terminated = truncated = False
    total_reward = 0
    step_count = 0
    
    while not (terminated or truncated) and step_count < MAX_EPISODE_STEPS:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        wrapped_env.render()
        
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

    # Define observation and action spaces for the coach
    observation_space = base_env.observation_space
    action_space = base_env.action_space


    # Create a coach agent
    coach_agent = CoachAgent(observation_space, action_space)

    # Wrap the environment with the coach
    #gymenv = InvertedPendulum3DEnvWithCoach(base_env, coach_agent, render_mode=None)
    gymenv = InvertedPendulum3DEnvWithCoach(base_env, coach_agent)

    if args.train:
        train(gymenv, args.sb3_algo)

    if args.test:
        if os.path.isfile(args.test):
            #gymenv = gym.make('InvertedPendulum3DWithCoach-v1', render_mode='human', max_episode_steps=MAX_EPISODE_STEPS)
            #gymenv = gym.make('InvertedPendulum3DWithCoach-v1', render_mode='human')
            base_env = gym.make('InvertedPendulum3D-v3', render_mode='human')
            #gymenv = InvertedPendulum3DEnvWithCoach(base_env, coach_agent, render_mode='human')
            coach_agent = CoachAgent(observation_space, action_space)
            gymenv = InvertedPendulum3DEnvWithCoach(base_env, coach_agent)
            test(gymenv, args.sb3_algo, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')