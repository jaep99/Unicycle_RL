import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict, Box
from gymnasium import Env
from gymnasium.wrappers import EnvCompatibility
from stable_baselines3 import SAC, TD3, A2C, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import os
import argparse
import custom_gym.envs.mujoco
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import datetime
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import json
import time

class PlottingCallback(BaseCallback):
    def __init__(self, wrapped_env, ax1, ax2, fig, verbose=0):
        super(PlottingCallback, self).__init__(verbose)
        self.wrapped_env = wrapped_env
        self.ax1 = ax1
        self.ax2 = ax2
        self.fig = fig
        self.last_episode_count = 0

        # Print to check the environment instance
        print(f"Environment in Callback: {id(self.wrapped_env)}", flush=True)
    
    def _on_step(self) -> bool:
        print(f"Current episodes list: {self.wrapped_env.episodes}", flush=True)  # Add this to debug
        # Check if a new episode is logged in the environment
        if len(self.wrapped_env.episodes) > self.last_episode_count:
            print("Calling update_plot...", flush=True)  # Add this for debugging
            self.last_episode_count = len(self.wrapped_env.episodes)
            # Update the plot with the latest data
            update_plot(self.wrapped_env, self.ax1, self.ax2, self.fig)
        return True


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
        #print(f"Step 4 - After environment step. Type of infos: {type(info)}, Content: {info}")
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
        self.model = PPO("MlpPolicy", self.coach_env, verbose=1, device="cpu")
        
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

    def __init__(self, env, coach_agent, render_mode=None, **kwargs):
        super(InvertedPendulum3DEnvWithCoach, self).__init__(env)
        self.coach_agent = coach_agent
        self._render_mode = render_mode
        # Move these outside of __init__ to ensure persistence between episodes
        if not hasattr(self, 'student_rewards'):
            self.student_rewards = []
        if not hasattr(self, 'coach_rewards'):
            self.coach_rewards = []
        if not hasattr(self, 'episodes'):
            self.episodes = []
        
        self.episode_student_rewards = []  # Store rewards for each episode
        self.episode_coach_rewards = []    # Store rewards for each episode
        #self.episode_coach_reward = 0
        self.episode_count = 0

        #self.current_episode_reward = 0  # Track the student's current episode reward
        #self.previous_episode_reward = 0  # Track the student's previous episode reward
        #self.coach_total_rewards = 0

    def reset(self, **kwargs):
        print(f"Before reset - Student Rewards Length: {len(self.student_rewards)}")
        print(f"Before reset - Coach Rewards Length: {len(self.coach_rewards)}")
        
        obs, info = self.env.reset(**kwargs)
        
        print(f"After reset - Student Rewards Length: {len(self.student_rewards)}")
        print(f"After reset - Coach Rewards Length: {len(self.coach_rewards)}")
        return obs, info


    def step(self, action):
        # Use env.unwrapped to access the base environment
        #observation = self.env.unwrapped._get_obs()
        observation = self.env._get_obs()
        
        # Get the coach action
        coach_action = self.coach_agent.select_action(observation, action)
        
        # Perform the environment step with the combined action
        combined_action = action + coach_action
        next_observation, student_reward, terminated, truncated, info = self.env.step(combined_action)
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        done = terminated or truncated

        # Calculate immediate reward feedback for the coach
        if len(self.episode_student_rewards) > 0:
            previous_student_reward = self.episode_student_rewards[-1]
            coach_reward = student_reward - previous_student_reward  # reward based on improvement
        else:
            coach_reward = student_reward  # first step uses student reward as base

        # Update the student's current episode cumulative reward
        #self.current_episode_reward += student_reward
        self.episode_student_rewards.append(student_reward)
        self.episode_coach_rewards.append(coach_reward)

        self.coach_agent.model.learn(total_timesteps=1)

        # Debugging statement to ensure coach_reward is correctly assigned
        print(f"Coach reward calculated: {coach_reward}")
        
        # Log transition for both student and coach
        #combined_obs = np.concatenate([observation, action])
        #combined_next_obs = np.concatenate([next_observation, combined_action])

        if done:
            # # Ensure coach_reward is initialized no matter the condition
            # if self.previous_episode_reward != 0:
            #     coach_reward = self.current_episode_reward - self.previous_episode_reward
            # else:
            #     coach_reward = self.current_episode_reward  # Handle initial episode case

            self.episode_count += 1
            self.student_rewards.append(np.mean(self.episode_student_rewards))
            self.coach_rewards.append(np.mean(self.episode_coach_rewards))
            self.episodes.append(self.episode_count)
            print(f"Episode Count: {self.episode_count}", flush=True)  # Debugging statement

            #self.previous_episode_reward = self.current_episode_reward
            #self.current_episode_reward = 0
            self.episode_student_rewards = []
            self.episode_coach_rewards = []
            #self.episode_coach_reward = 0

        return next_observation, student_reward, terminated, truncated, info
    
    def render(self):
        return self.env.render()


# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Max episode steps added
MAX_EPISODE_STEPS = 100


def train(env, sb3_algo):
    print("Entered train function", flush=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{sb3_algo}_{timestamp}"

    # Create a coach agent
    coach_agent = CoachAgent(env)
    print("Created CoachAgent", flush=True)

    # TensorBoard logging for custom rewards
    log_dir = f"logs/{run_name}_coach"
    logger = configure(log_dir, ["tensorboard"])
    print("TensorBoard logging configured", flush=True)

    # Wrap the environment with the coach
    wrapped_env = InvertedPendulum3DEnvWithCoach(env, coach_agent)
    print("Wrapped environment with CoachAgent", flush=True)
    print(f"Environment in Training Loop: {id(wrapped_env)}", flush=True)

    # Create seperated environment for model evaluation added
    #eval_env = gym.make('InvertedPendulum3D-v1', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)
    eval_env = InvertedPendulum3DEnvWithCoach(
        gym.make('InvertedPendulum3D-v5', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS),
        coach_agent
    )

    #eval_env = Monitor(eval_env)
    print("Created evaluation environment", flush=True)

    # EvalCallback added
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"{model_dir}/best_{run_name}",
                                 log_path=log_dir, eval_freq=10000,
                                 deterministic=True, render=False)
    
    print("EvalCallback added", flush=True)

    match sb3_algo:
        case 'PPO':
            model = PPO('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir)
        case 'SAC':
            model = SAC('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir)
        case 'TD3':
            model = TD3('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'A2C':
            model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case _:
            print('Algorithm not found')
            return

    print("Model initialized", flush=True)
    TIMESTEPS = 1000 
    iters = 0

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.show(block=False)  # Non-blocking show
    print("Matplotlib setup complete", flush=True)

    plotting_callback = PlottingCallback(wrapped_env, ax1, ax2, fig)

    #Combine both callbacks
    combined_callbacks = [eval_callback, plotting_callback]

    while True:
        try:
            iters += 1
            
            # Print out some debug info to ensure data is being collected
            print(f"Training iteration: {iters}", flush=True)
            print(f"Starting model training for {TIMESTEPS} timesteps", flush=True)

            # Train the student model
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=run_name, callback=combined_callbacks)
            
            # Print out some debug info post training
            print(f"Model training complete for iteration: {iters}", flush=True)
            
            model.save(f"{model_dir}/{run_name}_{TIMESTEPS*iters}")
            print(f"Model saved: {model_dir}/{run_name}_{TIMESTEPS*iters}", flush=True)

            # Print out some debug info to ensure data is being collected
            # print(f"Student Rewards Length: {len(wrapped_env.student_rewards)}", flush=True)
            # print(f"Coach Rewards Length: {len(wrapped_env.coach_rewards)}", flush=True)
            # print(f"Episode Length: {len(wrapped_env.episodes)}", flush=True)

        except Exception as e:
            print(f"Error in training loop: {e}", flush=True)
                                             
    plt.ioff()
    plt.show()
    

def update_plot(wrapped_env, ax1, ax2, fig):
    # # Ensure there is enough data to plot
    # if len(wrapped_env.student_rewards) < 1 or len(wrapped_env.episodes) < 1:
    #     print("Not enough data to plot yet.")
    #     print(f"Current Student Rewards Length: {len(wrapped_env.student_rewards)}")
    #     print(f"Current Episode Length: {len(wrapped_env.episodes)}")
    #     return  # Skip plotting if not enough data

    # print("Updating plot with data:", flush=True)
    # print(f"Student Rewards: {wrapped_env.student_rewards[-5:]}")  # Check last few entries
    # print(f"Coach Rewards: {wrapped_env.coach_rewards[-5:]}")      # Check last few entries
    # print(f"Episodes: {wrapped_env.episodes[-5:]}")

    # # # Remove duplicates and sort data
    # # unique_timesteps, unique_indices = np.unique(wrapped_env.timesteps, return_index=True)
    # # unique_student_rewards = np.array(wrapped_env.student_rewards)[unique_indices]
    # # unique_coach_rewards = np.array(wrapped_env.coach_rewards)[unique_indices]

    # # Clear previous data
    # ax1.clear()
    # ax2.clear()

    # # Plot using episode mean data
    # ax1.plot(wrapped_env.episodes, wrapped_env.student_rewards, label='Student Reward Mean')
    # ax1.set_xlabel('Episode')
    # ax1.set_ylabel('Mean Reward')
    # ax1.set_title('Student Mean Reward over Episodes')
    # ax1.legend()

    # ax2.plot(wrapped_env.episodes, wrapped_env.coach_rewards, label='Coach Reward Mean', color='r')
    # ax2.set_xlabel('Episode')
    # ax2.set_ylabel('Mean Reward')
    # ax2.set_title('Coach Mean Reward over Episodes')
    # ax2.legend()

    # # Redraw the plots
    # # Force update of the canvas
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    # plt.pause(0.1) 

    print("Updating plot...", flush=True)

    # Simplify the check to ensure it's always called
    ax1.plot(wrapped_env.episodes, wrapped_env.student_rewards, 'bo-', label='Student Reward Mean')
    ax2.plot(wrapped_env.episodes, wrapped_env.coach_rewards, 'ro-', label='Coach Reward Mean')

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)


# def update_plot(wrapped_env, ax1, ax2, fig):
#     # Ensure there is enough data to plot
#     if len(wrapped_env.student_rewards) < 1 or len(wrapped_env.episodes) < 1:
#         print("Not enough data to plot yet.")
#         print(f"Current Student Rewards Length: {len(wrapped_env.student_rewards)}")
#         print(f"Current Episode Length: {len(wrapped_env.episodes)}")
#         return  # Skip plotting if not enough data

#     print("Updating plot with new data:", flush=True)
    
#     # Extract the latest data point
#     latest_student_reward = wrapped_env.student_rewards[-1]
#     latest_coach_reward = wrapped_env.coach_rewards[-1]
#     latest_episode = wrapped_env.episodes[-1]

#     print(f"New Student Reward: {latest_student_reward}")
#     print(f"New Coach Reward: {latest_coach_reward}")
#     print(f"Episode: {latest_episode}")

#     # Plot the new data point without clearing the previous plot
#     ax1.plot(latest_episode, latest_student_reward, 'bo', label='Student Reward Mean' if latest_episode == 1 else "")  # Blue dots
#     ax2.plot(latest_episode, latest_coach_reward, 'ro', label='Coach Reward Mean' if latest_episode == 1 else "")  # Red dots

#     # Redraw the legend for the first plot if labels were added
#     if latest_episode == 1:
#         ax1.legend()
#         ax2.legend()

#     # Force update of the canvas with the new data points
#     fig.canvas.draw()
#     fig.canvas.flush_events()
#     plt.pause(0.1)  # This ensures the plot is updated



def test(env, sb3_algo, path_to_model):
    observation_space = env.observation_space
    action_space = env.action_space
    coach_agent = CoachAgent(env)
    gymenv = InvertedPendulum3DEnvWithCoach(env, coach_agent, logger=None)

    match sb3_algo:
        case 'PPO':
            model = PPO.load(path_to_model, env =gymenv)
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

    # Plot the rewards obtained during testing
    update_plot(gymenv)

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    #gymenv = gym.make('InvertedPendulum3DWithCoach-v1', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)
    #gymenv = gym.make('InvertedPendulum3DWithCoach-v1', render_mode=None)

    #base_env = EnvCompatibility(base_env)

    # Wrap the environment with the coach
    #gymenv = InvertedPendulum3DEnvWithCoach(base_env, coach_agent, render_mode=None)
    #gymenv = InvertedPendulum3DEnvWithCoach(base_env, coach_agent)

    if args.train:
        # TensorBoard logging for training
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.sb3_algo}_{timestamp}"

        base_env = gym.make('InvertedPendulum3D-v5', render_mode=None)
        check_env(base_env)

        # Create a coach agent
        coach_agent = CoachAgent(base_env)

        # Wrap the environment with the coach and logger
        gymenv = InvertedPendulum3DEnvWithCoach(base_env, coach_agent)

        train(gymenv, args.sb3_algo)

    if args.test:
        if os.path.isfile(args.test):
            #gymenv = gym.make('InvertedPendulum3DWithCoach-v1', render_mode='human', max_episode_steps=MAX_EPISODE_STEPS)
            #gymenv = gym.make('InvertedPendulum3DWithCoach-v1', render_mode='human')
            test_env = gym.make('InvertedPendulum3D-v5', render_mode='human')
            check_env(test_env)
            #gymenv = InvertedPendulum3DEnvWithCoach(base_env, coach_agent, render_mode='human')
            # Unwrap the environment if it's wrapped with any wrapper like TimeLimit, OrderEnforcing, or PassiveEnvChecker
            coach_agent = CoachAgent(test_env)
            gymenv = InvertedPendulum3DEnvWithCoach(test_env, coach_agent)
            test(gymenv, args.sb3_algo, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')