import gymnasium as gym
from stable_baselines3 import SAC
import os
import argparse
import custom_gym.envs.mujoco
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import time

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

"""
## Student agent training code for the Unicycle Project.
## Creating the ideal solution model will be done in this code.

** Code Summary **
    - unicycle_pendulum_trajectory_rl_v1.py (training)
    - unicycle_pendulum_trajectory_3d_v0.py (environment)
"""

# Maximum number of steps per episode
MAX_EPISODE_STEPS = 10000

# Success thresholds for model evaluation and saving: 1000 (ideal)
SUCCESS_THRESHOLDS = [1000]

class UnicyclePositionLogger(BaseCallback):
    """
    Custom callback for logging unicycle positions, actions, and success information during training.
    """
    def __init__(self, best_model_path, verbose=0):
        super(UnicyclePositionLogger, self).__init__(verbose)
        self.best_model_path = best_model_path
        self.unicycle_positions = []
        self.actions = []
        self.timesteps = []
        self.success_timesteps = []
        self.success_counts = []
        self.total_episodes = []
        self.success_rates = []
        self.avg_rewards = []
        self.avg_goal_times = []
        self.training_times = []
        self.start_time = time.time()

        # Progress update variables
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_counts = []
        self.current_episode = 0
        self.current_episode_reward = 0

    def _on_step(self):
        """
        This method is called at each step of the training process.
        It logs various metrics for later analysis and plotting.
        """
        obs = self.locals['new_obs'][0]
        info = self.locals['infos'][0]
        self.current_episode_reward += info['reward']
        
        
        self.unicycle_positions.append(obs[:3])  # Log unicycle position
        self.actions.append(self.locals['actions'][0])  # Log action taken
        self.timesteps.append(self.num_timesteps)  # Log current timestep
        
        self.success_counts.append(info['success_count'])
        self.total_episodes.append(info['total_episodes'])
        
        # Calculate success rate over last 100 episodes
        recent_successes = sum([1 for s in self.success_counts[-100:] if s > 0])
        self.success_rates.append(recent_successes / min(100, len(self.success_counts)))
        
        # Calculate average reward over last 100 episodes
        self.avg_rewards.append(np.mean([info['reward'] for info in self.locals['infos'][-100:]]))
        
        # Calculate average goal reach time
        if info['goal_reached']:
            self.avg_goal_times.append(info['steps'])
        else:
            self.avg_goal_times.append(self.avg_goal_times[-1] if self.avg_goal_times else 0)
        
        self.training_times.append(time.time() - self.start_time)

        # Plot progress every 1000 episodes
        if info.get('terminal_observation') is not None:
            self.current_episode += 1
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_successes.append(info['success_count'])
            self.episode_counts.append(self.current_episode)
            self.current_episode_reward = 0
            
            if self.current_episode % 1000 == 0:
                self.plot_progress()
        
        return True
    
    def plot_progress(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.episode_counts, self.episode_successes, 'b.-')
        plt.title('Cumulative Successes over Episodes')
        plt.xlabel('Episodes')
        plt.ylabel('Cumulative Successes')
        plt.grid(True)
        plt.savefig(f'{self.best_model_path}/progress_{self.current_episode}.png')
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.plot(self.episode_counts, self.episode_rewards, 'r.-')
        plt.title('Episode Rewards over Time')
        plt.xlabel('Episodes')
        plt.ylabel('Episode Reward')
        plt.grid(True)
        plt.savefig(f'{self.best_model_path}/rewards_{self.current_episode}.png')
        plt.close()

def plot_training_results(logger, run_name):
    """
    Function to plot various training metrics.
    """
    fig, axs = plt.subplots(3, 2, figsize=(20, 30))
    fig.suptitle(f'Training Results for {run_name}', fontsize=16)

    # 1. Success rate vs episodes graph
    axs[0, 0].plot(logger.total_episodes, logger.success_rates)
    axs[0, 0].set_xlabel('Total Episodes')
    axs[0, 0].set_ylabel('Success Rate')
    axs[0, 0].set_title('Success Rate vs Episodes')

    # 2. Average reward vs episodes graph
    axs[0, 1].plot(logger.total_episodes, logger.avg_rewards)
    axs[0, 1].set_xlabel('Total Episodes')
    axs[0, 1].set_ylabel('Average Reward')
    axs[0, 1].set_title('Average Reward vs Episodes')

    # 3. Goal reach time vs success count graph
    axs[1, 0].plot(logger.success_counts, logger.avg_goal_times)
    axs[1, 0].set_xlabel('Cumulative Successes')
    axs[1, 0].set_ylabel('Average Goal Reach Time (steps)')
    axs[1, 0].set_title('Goal Reach Time vs Successes')

    # 4. Training time vs success count graph
    axs[1, 1].plot(logger.success_counts, logger.training_times)
    axs[1, 1].set_xlabel('Cumulative Successes')
    axs[1, 1].set_ylabel('Total Training Time (seconds)')
    axs[1, 1].set_title('Training Time vs Successes')

    # 5. Performance metrics comparison bar graph
    thresholds = ['10^2', '10^3', '10^4']
    metrics = ['Avg Reward', 'Avg Goal Time', 'Success Rate']
    data = np.array([
        [logger.avg_rewards[i] for i in [100, 1000, 10000]],
        [logger.avg_goal_times[i] for i in [100, 1000, 10000]],
        [logger.success_rates[i] for i in [100, 1000, 10000]]
    ])

    x = np.arange(len(thresholds))
    width = 0.25

    for i in range(len(metrics)):
        axs[2, 0].bar(x + i*width, data[i], width, label=metrics[i])

    axs[2, 0].set_xlabel('Success Thresholds')
    axs[2, 0].set_ylabel('Performance Metrics')
    axs[2, 0].set_title('Performance Metrics Comparison')
    axs[2, 0].set_xticks(x + width)
    axs[2, 0].set_xticklabels(thresholds)
    axs[2, 0].legend()

    # Highlighting 10^2, 10^3, 10^4 points in relevant graphs
    for ax in [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]:
        for threshold in [100, 1000, 10000]:
            ax.axvline(x=threshold, color='r', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{logger.best_model_path}/training_results.png')
    plt.close()

def train(env):
    """
    Main training function for the SAC agent.
    """
    # Generate a unique timestamp for this training run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"SAC_{timestamp}"
    
    # Create a directory to save the best models
    best_model_path = f"{model_dir}/best_{run_name}"
    os.makedirs(best_model_path, exist_ok=True)
    
    # Create an evaluation environment
    eval_env = gym.make('UnicyclePendulumTrajectory-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)
    
    # Set up the evaluation callback
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=best_model_path,
        log_path=log_dir, 
        eval_freq=10000,
        deterministic=True, 
        render=False,
    )

    # Set up the custom logger callback
    unicycle_logger = UnicyclePositionLogger(best_model_path)
    
    # Combine callbacks
    callbacks = [eval_callback, unicycle_logger]

    # Initialize the SAC model
    model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)

    start_time = time.time()
    threshold_index = 0

    # Main training loop
    total_episodes = 0
    current_success_count = 0
    while threshold_index < len(SUCCESS_THRESHOLDS):
        # Train the model for 10000 timesteps
        model.learn(total_timesteps=10000, reset_num_timesteps=False, tb_log_name=run_name, callback=callbacks)
        
        # Get the latest info from the environment
        obs, info = env.reset()
        current_success_count = info.get('success_count', current_success_count)
        total_episodes = info.get('total_episodes', total_episodes + 1)
        
        print(f"Total episodes: {total_episodes}, Total Successes: {current_success_count}")
        
        # Check if we've reached the 100 episodes
        if total_episodes % 100 == 0:
            unicycle_logger.plot_progress()

        """
        # Check if we've reached the next success threshold
        if current_success_count >= SUCCESS_THRESHOLDS[threshold_index]:
            print(f"\nReached {SUCCESS_THRESHOLDS[threshold_index]} successes!")
            model.save(f"{best_model_path}/model_{SUCCESS_THRESHOLDS[threshold_index]}_successes")
            threshold_index += 1
        """

        # Check if we've reached threshold (1000)
        # 1000 success MAX
        if current_success_count >= SUCCESS_THRESHOLDS[-1]:
            end_time = time.time()
            training_time = end_time - start_time
            print(f"\nTraining completed! All success thresholds achieved.")
            print(f"Total training time: {training_time:.2f} seconds")
            print(f"Total episodes: {total_episodes}")
            break

    # Save the final model
    model.save(f"{model_dir}/{run_name}_final")
    
    # Generate and save the training results plots
    plot_training_results(unicycle_logger, run_name)

def test(env, path_to_model):
    """
    Function to test a trained model.
    """
    # Load the trained model
    model = SAC.load(path_to_model, env=env)

    obs, info = env.reset()
    terminated = truncated = False
    total_reward = 0
    step_count = 0
    
    # Main testing loop
    while not (terminated or truncated) and step_count < MAX_EPISODE_STEPS:
        # Get the model's prediction and take a step in the environment
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        total_reward += reward
        step_count += 1
        
        # Extract relevant information from the observation
        unicycle_pos = obs[:3]
        unicycle_quat = obs[3:7]
        pendulum_quat = obs[7:11]
        
        # Convert quaternions to Euler angles
        unicycle_euler = Rotation.from_quat(np.roll(unicycle_quat, -1)).as_euler('xyz', degrees=False)
        pendulum_euler = Rotation.from_quat(np.roll(pendulum_quat, -1)).as_euler('xyz', degrees=False)
        
        wheel_velocity = obs[21]  # Assuming wheel velocity is at index 21

        # Print step information
        print(f"Step: {step_count}, Reward: {reward:.4f}")
        print(f"Unicycle Position: ({unicycle_pos[0]:.4f}, {unicycle_pos[1]:.4f}, {unicycle_pos[2]:.4f})")
        print(f"Unicycle Roll, Pitch, Yaw: ({unicycle_euler[0]:.4f}, {unicycle_euler[1]:.4f}, {unicycle_euler[2]:.4f})")
        print(f"Pendulum Roll, Pitch, Yaw: ({pendulum_euler[0]:.4f}, {pendulum_euler[1]:.4f}, {pendulum_euler[2]:.4f})")
        print(f"Wheel Velocity: {wheel_velocity:.4f}")
        print(f"Success Count: {info['success_count']}")
        print("------------------------------")

    # Print final episode statistics
    print(f"\nEpisode finished after {step_count} steps")
    print(f"Total reward: {total_reward}")
    print(f"Average reward per step: {total_reward / step_count}")
    print(f"Total successes: {info['success_count']}")
    
    if info.get('goal_reached', False):
        print("Episode ended by reaching the goal!")
    elif terminated:
        print("Episode ended by termination condition.")
    elif truncated:
        print("Episode ended by truncation (max steps reached).")

if __name__ == '__main__':
    # Set up command line argument parsing₩
    parser = argparse.ArgumentParser(description='Train or test SAC model.')
    parser.add_argument('-train', '--train', action='store_true')
    parser.add_argument('-test', '--test', metavar='path_to_model')
    args = parser.parse_args()

    # Create the unicycle environment
    env = gym.make('UnicyclePendulumTrajectory-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)

    if args.train:
        train(env)

    if args.test:
        if os.path.isfile(args.test):
            test_env = gym.make('UnicyclePendulumTrajectory-v0', render_mode='human', max_episode_steps=MAX_EPISODE_STEPS)
            test(test_env, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')