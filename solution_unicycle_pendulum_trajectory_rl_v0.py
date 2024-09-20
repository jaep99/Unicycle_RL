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

# Maximum number of steps per episode
MAX_EPISODE_STEPS = 10000

# Custom callback to log unicycle and pendulum positions and orientations
class UnicyclePendulumLogger(BaseCallback):
    def __init__(self, verbose=0):
        super(UnicyclePendulumLogger, self).__init__(verbose)
        self.unicycle_positions = []
        self.unicycle_roll_pitch_yaw = []
        self.pendulum_roll_pitch_yaw = []
        self.wheel_velocities = []
        self.timesteps = []
        self.success_counts = []

    def _on_step(self) -> bool:
        obs = self.locals['new_obs'][0]
        info = self.locals['infos'][0]
        self.unicycle_positions.append(obs[:3])
        
        unicycle_quat = obs[3:7]
        pendulum_quat = obs[7:11]
        
        unicycle_euler = Rotation.from_quat(np.roll(unicycle_quat, -1)).as_euler('xyz')
        pendulum_euler = Rotation.from_quat(np.roll(pendulum_quat, -1)).as_euler('xyz')
        
        self.unicycle_roll_pitch_yaw.append(unicycle_euler)
        self.pendulum_roll_pitch_yaw.append(pendulum_euler)
        self.wheel_velocities.append(obs[21])
        self.timesteps.append(self.num_timesteps)
        self.success_counts.append(info['success_count'])
        return True

# Environment wrapper to add bonus reward
class RewardWrapper(gym.Wrapper):
    def __init__(self, env, bonus_reward=100.0):
        super().__init__(env)
        self.bonus_reward = bonus_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get('goal_reached', False):
            reward += self.bonus_reward
            print("Goal reached!")  # This will print during both training and testing
        return obs, reward, terminated, truncated, info

# Function to plot unicycle and pendulum positions and orientations
def plot_unicycle_pendulum_position(logger, run_name):
    fig = plt.figure(figsize=(20, 15))
    
    # 3D plot
    ax = fig.add_subplot(221, projection='3d')
    positions = np.array(logger.unicycle_positions)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.set_title(f'Unicycle 3D Movement')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    
    # Top-down view
    ax = fig.add_subplot(222)
    ax.plot(positions[:, 0], positions[:, 1])
    ax.set_title(f'Unicycle Movement (Top-down view)')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.axis('equal')
    
    # Unicycle and Pendulum roll, pitch and yaw
    ax = fig.add_subplot(223)
    unicycle_roll_pitch_yaw = np.array(logger.unicycle_roll_pitch_yaw)
    pendulum_roll_pitch_yaw = np.array(logger.pendulum_roll_pitch_yaw)
    ax.plot(logger.timesteps, unicycle_roll_pitch_yaw, linestyle='-')
    ax.plot(logger.timesteps, pendulum_roll_pitch_yaw, linestyle='--')
    ax.set_title('Unicycle and Pendulum Roll, Pitch and Yaw')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Angle (radians)')
    ax.legend(['Unicycle Roll', 'Unicycle Pitch', 'Unicycle Yaw', 
               'Pendulum Roll', 'Pendulum Pitch', 'Pendulum Yaw'])

    # Success count
    ax = fig.add_subplot(224)
    ax.plot(logger.timesteps, logger.success_counts)
    ax.set_title('Cumulative Success Count')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Number of Successes')

    plt.tight_layout()
    
    save_path = os.path.join(log_dir, f'{run_name}_analysis.png')
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

# Main training function
def train(env):
    # Generate a unique timestamp for this training run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"SAC_{timestamp}"
    
    # Set the path for saving the best model
    best_model_path = f"{model_dir}/best_{run_name}"
    os.makedirs(best_model_path, exist_ok=True)
    
    # Create a separate environment for model evaluation
    eval_env = RewardWrapper(gym.make('SolutionUnicyclePendulumTrajectory-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS))
    
    # Initialize the EvalCallback for saving the best model
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=best_model_path,
        log_path=log_dir, 
        eval_freq=10000,
        deterministic=True, 
        render=False,
    )

    # Initialize the UnicyclePendulumLogger
    unicycle_pendulum_logger = UnicyclePendulumLogger()
    
    # Combine callbacks
    callbacks = [eval_callback, unicycle_pendulum_logger]

    # Initialize the SAC model
    model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)

    # Train the model until 100 successes are achieved
    total_timesteps = 0
    start_time = time.time()
    prev_success_count = 0

    while True:
        model.learn(total_timesteps=10000, reset_num_timesteps=False, tb_log_name=run_name, callback=callbacks)
        total_timesteps += 10000
        
        current_success_count = unicycle_pendulum_logger.success_counts[-1] if unicycle_pendulum_logger.success_counts else 0
        new_successes = current_success_count - prev_success_count
        
        print(f"Total timesteps: {total_timesteps}, Total Successes: {current_success_count}, New Successes: {new_successes}")
        
        if new_successes > 0:
            print(f"Goal reached! New success count: {new_successes}")
        
        prev_success_count = current_success_count

        if current_success_count >= 100:
            end_time = time.time()
            training_time = end_time - start_time
            print(f"\nTraining completed! 100 successes achieved.")
            print(f"Total training time: {training_time:.2f} seconds")
            print(f"Total timesteps: {total_timesteps}")
            break

    # Save the final model
    model.save(f"{model_dir}/{run_name}_final")

    # Generate and save the unicycle and pendulum position plot
    plot_unicycle_pendulum_position(unicycle_pendulum_logger, run_name)

# Function to test the trained model
def test(env, path_to_model):
    # Load the SAC model
    model = SAC.load(path_to_model, env=env)

    obs, _ = env.reset()
    terminated = truncated = False
    total_reward = 0
    step_count = 0
    success_count = 0
    
    # Run the episode
    while not (terminated or truncated) and step_count < MAX_EPISODE_STEPS:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        total_reward += reward
        step_count += 1
        
        unicycle_pos = obs[:3]
        unicycle_quat = obs[3:7]
        pendulum_quat = obs[7:11]
        
        unicycle_euler = Rotation.from_quat(np.roll(unicycle_quat, -1)).as_euler('xyz', degrees=False)
        pendulum_euler = Rotation.from_quat(np.roll(pendulum_quat, -1)).as_euler('xyz', degrees=False)
        
        wheel_velocity = obs[21]  # Assuming wheel velocity is at index 21

        print(f"Step: {step_count}, Reward: {reward:.4f}")
        print(f"Unicycle Position: ({unicycle_pos[0]:.4f}, {unicycle_pos[1]:.4f}, {unicycle_pos[2]:.4f})")
        print(f"Unicycle Roll, Pitch, Yaw: ({unicycle_euler[0]:.4f}, {unicycle_euler[1]:.4f}, {unicycle_euler[2]:.4f})")
        print(f"Pendulum Roll, Pitch, Yaw: ({pendulum_euler[0]:.4f}, {pendulum_euler[1]:.4f}, {pendulum_euler[2]:.4f})")
        print(f"Wheel Velocity: {wheel_velocity:.4f}")
        print(f"Success Count: {info['success_count']}")
        print("------------------------------")

        if info.get('goal_reached', False):
            success_count += 1
    
    # Print episode summary
    print(f"\nEpisode finished after {step_count} steps")
    print(f"Total reward: {total_reward}")
    print(f"Average reward per step: {total_reward / step_count}")
    print(f"Total successes: {success_count}")
    
    if info.get('goal_reached', False):
        print("Episode ended by reaching the goal!")
    elif terminated:
        print("Episode ended by termination condition.")
    elif truncated:
        print("Episode ended by truncation (max steps reached).")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or test SAC model.')
    parser.add_argument('-train', '--train', action='store_true')
    parser.add_argument('-test', '--test', metavar='path_to_model')
    args = parser.parse_args()

    # Create and wrap the environment
    gymenv = RewardWrapper(gym.make('SolutionUnicyclePendulumTrajectory-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS))

    if args.train:
        train(gymenv)

    if args.test:
        if os.path.isfile(args.test):
            # For testing, use render_mode='human'
            test_env = RewardWrapper(gym.make('SolutionUnicyclePendulumTrajectory-v0', render_mode='human', max_episode_steps=MAX_EPISODE_STEPS))
            test(test_env, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')