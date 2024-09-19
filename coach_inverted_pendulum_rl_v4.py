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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import datetime
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import json
import time
import threading
import time

"""
class PlottingCallback(BaseCallback):
    def __init__(self, wrapped_env, ax1, ax2, fig, update_freq=1000, verbose=0):
        super(PlottingCallback, self).__init__(verbose)
        self.wrapped_env = wrapped_env
        self.ax1 = ax1
        self.ax2 = ax2
        self.fig = fig
        self.update_freq = update_freq

        # Print to check the environment instance
        print(f"Environment in Callback: {id(self.wrapped_env)}", flush=True)
    
    def _on_step(self) -> bool: 
       return True
    
    def _on_rollout_end(self) -> None:
        # Ensure rewards have been collected before trying to plot
        if len(self.wrapped_env.student_rewards) > 0:
            print(f"Student Rewards in Callback: {self.wrapped_env.student_rewards}")
            print(f"Coach Rewards in Callback: {self.wrapped_env.coach_rewards}")
            update_plot(self.wrapped_env, self.ax1, self.ax2, self.fig)
        else:
            print("No data to plot yet.")
        plt.pause(0.01)
"""        

class PendulumCoachLogger(BaseCallback):
    def __init__(self, best_model_path, verbose=0):
        super(PendulumCoachLogger, self).__init__(verbose)
        self.best_model_path = best_model_path
        self.rewards = []
        self.timesteps = []
        self.iteration_rewards = []

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        self.rewards.append(reward)
        self.timesteps.append(self.num_timesteps)

        # Save rewards and log every 3000 timesteps
        if self.num_timesteps % 3000 == 0:
            self.iteration_rewards.append(self.rewards[-3000:])
            self.plot_graphs()
        
        return True

    def plot_graphs(self):
        iteration = self.num_timesteps // 3000
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot the cumulative rewards
        cumulative_rewards = np.cumsum(self.rewards)
        ax.plot(self.timesteps, cumulative_rewards, label='Cumulative Reward')
        ax.set_title('Cumulative Rewards over Time')
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Cumulative Reward')
        ax.legend()
        
        # Save the figure
        save_path = os.path.join(self.best_model_path, f'pendulum_coach_analysis_iteration_{iteration}.png')
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
        plt.close()

    def plot_final_reward_graph(self):
        plt.figure(figsize=(10, 5))
        iteration_rewards = [np.sum(self.rewards[i:i+3000]) for i in range(0, len(self.rewards), 3000)]
        plt.plot(range(1, len(iteration_rewards) + 1), iteration_rewards)
        plt.title('Total Rewards per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Total Reward')
        save_path = os.path.join(self.best_model_path, 'final_reward_graph.png')
        plt.savefig(save_path)
        print(f"Final reward graph saved to: {save_path}")
        plt.close()

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

    def __init__(self, env, coach_agent, render_mode=None, **kwargs):
        super(InvertedPendulum3DEnvWithCoach, self).__init__(env)
        self.coach_agent = coach_agent
        self._render_mode = render_mode
        self.student_rewards = []
        self.coach_rewards = []
        self.timesteps = []

        self.current_episode_reward = 0  # Track the student's current episode reward
        self.previous_episode_reward = None  # Track the student's previous episode reward
        self.coach_total_rewards = 0

    def reset(self, **kwargs):
        # Reset the wrapped environment (InvertedPendulum3DEnv)
        #print(f"Before reset - Student Rewards: {self.student_rewards}")
        obs, info = self.env.reset(**kwargs)
        #print(f"After reset - Student Rewards: {self.student_rewards}")
        
        # Reset the current episode reward
        self.step_count = 0
        self.current_episode_reward = 0
        self.coach_total_rewards = 0
        self.coach_rewards = []
        # Optional: Print statements to check
        print(f"Reset called. Student Rewards Length: {len(self.student_rewards)}")
        print(f"Reset called. Coach Rewards Length: {len(self.coach_rewards)}")
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

        improvement = 0
        # Log coach reward
        if self.previous_episode_reward is not None:
            improvement = self.current_episode_reward - self.previous_episode_reward
            coach_reward = improvement
        else:
            coach_reward = 0 # No reward for the first episode

        print(f"Current Episode Reward: {self.current_episode_reward}")
        print(f"Previous Episode Reward: {self.previous_episode_reward}")
        print(f"Improvement: {improvement}")
        print(f"Coach Reward: {coach_reward}")

        
        self.student_rewards.append(student_reward)
        self.coach_rewards.append(coach_reward)
        self.timesteps.append(self.env.unwrapped.step_count)
        # Storing coach reward for this step
        self.coach_total_rewards += coach_reward

        # Print debug information
        #print(f"Step: {self.env.unwrapped.step_count}, Student Reward: {student_reward}, Coach Reward: {coach_reward}")
        # print(f"Step executed. Student Reward: {student_reward}, Coach Reward: {coach_reward}")
        # print(f"Current Student Rewards: {self.student_rewards}")
        #print(f"Current Coach Rewards: {self.coach_rewards}")

        # Print debug information
        #print(f"Step {self.env.unwrapped.step_count} - Student Reward: {student_reward}, Coach Reward: {coach_reward}")
        #print(f"Total Student Rewards Collected: {len(self.student_rewards)}, Total Coach Rewards Collected: {len(self.coach_rewards)}")
        print(f"Is it termined?????: {done}")
        if done:
            # Create combined observation for replay buffer (observation + action)
            #combined_observation = np.concatenate([observation, action])
            #combined_next_observation = np.concatenate([next_observation, action])

            """
            # Log the coach reward and train the coach
            self.coach_agent.model.replay_buffer.add(
                combined_observation, # Comb observation
                combined_next_observation, # Next observation
                combined_action, # Combined action
                np.array([coach_reward]), # Coach reward
                np.array([done]), # Episode termination status
                [info] # Additional info
            )
            """

            # Update the previous episode reward
            self.previous_episode_reward = self.current_episode_reward
            self.current_episode_reward = 0  # Reset for the next episode
            #self.coach_total_rewards = 0
        
        return next_observation, student_reward, terminated, truncated, info
    
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


def train(env, sb3_algo, plot_update_interval=1000):
    print("Entered train function", flush=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{sb3_algo}_{timestamp}"

    best_model_path = f"{model_dir}/best_{run_name}"
    os.makedirs(best_model_path, exist_ok=True)

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
        gym.make('InvertedPendulum3D-v3', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS),
        coach_agent
    )

    eval_env = Monitor(eval_env)
    print("Created evaluation environment", flush=True)

    # EvalCallback added
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"{model_dir}/best_{run_name}",
                                 log_path=log_dir, eval_freq=10000,
                                 deterministic=True, render=False)
    
    print("EvalCallback added", flush=True)

    match sb3_algo:
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
    TIMESTEPS = 3000
    total_timesteps = 0
    max_timesteps = 3000 * 100

    pendulum_coach_logger = PendulumCoachLogger(best_model_path)
    callbacks = [eval_callback, pendulum_coach_logger]


    while total_timesteps < max_timesteps:

        # Train the student model using a batch of timesteps
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=run_name, callback=callbacks)


        # Save the model after each batch of timesteps
        model.save(f"{model_dir}/{run_name}_{total_timesteps}")

        total_timesteps += TIMESTEPS
        pendulum_coach_logger.plot_graphs()
        # Train the coach agent after each batch
        coach_agent.train(total_timesteps=total_timesteps)

    print("Training completed after reaching 300000 timesteps.")
    pendulum_coach_logger.plot_final_reward_graph()  # Final reward graph
        

    

def update_plot(wrapped_env, ax1, ax2, fig):
# Clear previous data from the plots
    ax1.clear()
    ax2.clear()
    
    # Print debug info about data to be plotted
    print("Updating plot with data:", flush=True)
    print(f"Student Rewards: {wrapped_env.student_rewards}", flush=True)
    print(f"Coach Rewards: {wrapped_env.coach_rewards}", flush=True)
    print(f"Timesteps: {wrapped_env.timesteps}", flush=True)
    
    ax1.plot(wrapped_env.timesteps, wrapped_env.student_rewards, label='Student Reward')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Reward')
    ax1.set_title('Student Reward over Time')
    ax1.legend()

    # Plot Coach rewards
    ax2.plot(wrapped_env.timesteps, wrapped_env.coach_rewards, label='Coach Reward', color='r')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Reward')
    ax2.set_title('Coach Reward over Time')
    ax2.legend()

    # Redraw the plots
    # Force update of the canvas
    fig.canvas.draw()
    fig.canvas.flush_events()

    plt.pause(0.5)  # Adjust pause time if necessary


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
    parser.add_argument('--plot_update_interval', type=int, default=1000, help='Interval to update the plot during training')
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

        base_env = gym.make('InvertedPendulum3D-v3', render_mode=None)
        check_env(base_env)

        # Create a coach agent
        coach_agent = CoachAgent(base_env)

        # Wrap the environment with the coach and logger
        gymenv = InvertedPendulum3DEnvWithCoach(base_env, coach_agent)

        train(gymenv, args.sb3_algo, plot_update_interval=args.plot_update_interval)

    if args.test:
        if os.path.isfile(args.test):
            #gymenv = gym.make('InvertedPendulum3DWithCoach-v1', render_mode='human', max_episode_steps=MAX_EPISODE_STEPS)
            #gymenv = gym.make('InvertedPendulum3DWithCoach-v1', render_mode='human')
            test_env = gym.make('InvertedPendulum3D-v3', render_mode='human')
            check_env(test_env)
            #gymenv = InvertedPendulum3DEnvWithCoach(base_env, coach_agent, render_mode='human')
            # Unwrap the environment if it's wrapped with any wrapper like TimeLimit, OrderEnforcing, or PassiveEnvChecker
            coach_agent = CoachAgent(test_env)
            gymenv = InvertedPendulum3DEnvWithCoach(test_env, coach_agent)
            test(gymenv, args.sb3_algo, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')