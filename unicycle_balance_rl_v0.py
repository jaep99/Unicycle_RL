import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C
import os
import argparse
import custom_gym.envs.mujoco
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Maximum number of steps per episode
MAX_EPISODE_STEPS = 30000

# Custom callback to log unicycle positions and orientations
class UnicyclePositionLogger(BaseCallback):
    def __init__(self, best_model_path, verbose=0):
        super(UnicyclePositionLogger, self).__init__(verbose)
        self.best_model_path = best_model_path
        self.unicycle_positions = []
        self.unicycle_roll_pitch = []
        self.wheel_velocities = []
        self.timesteps = []

    def _on_step(self) -> bool:
        obs = self.locals['new_obs'][0]
        self.unicycle_positions.append(obs[:3])
        
        quat_mujoco = obs[3:7]
        quat_scipy = np.roll(quat_mujoco, -1)  # Convert to SciPy quaternion order (xyzw)
        euler = Rotation.from_quat(quat_scipy).as_euler('xyz')
        
        self.unicycle_roll_pitch.append(euler[:2])
        self.wheel_velocities.append(obs[14])  # Assuming wheel velocity is at index 14
        self.timesteps.append(self.num_timesteps)
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

# Function to plot unicycle positions and orientations
def plot_unicycle_position(logger, run_name, iteration):
    fig = plt.figure(figsize=(20, 15))
    
    # 3D plot
    ax = fig.add_subplot(221, projection='3d')
    positions = np.array(logger.unicycle_positions)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.set_title(f'Unicycle 3D Movement (Iteration {iteration})')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    
    # Top-down view
    ax = fig.add_subplot(222)
    ax.plot(positions[:, 0], positions[:, 1])
    ax.set_title(f'Unicycle Movement (Top-down view, Iteration {iteration})')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.axis('equal')
    
    # Unicycle roll and pitch
    ax = fig.add_subplot(223)
    unicycle_roll_pitch = np.array(logger.unicycle_roll_pitch)
    ax.plot(logger.timesteps, unicycle_roll_pitch)
    ax.set_title('Unicycle Roll and Pitch')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Angle (radians)')
    ax.legend(['Roll', 'Pitch'])

    # Wheel velocity
    ax = fig.add_subplot(224)
    ax.plot(logger.timesteps, logger.wheel_velocities)
    ax.set_title('Wheel Velocity')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Angular Velocity (rad/s)')

    plt.tight_layout()
    
    save_path = os.path.join(logger.best_model_path, f'unicycle_analysis_iter_{iteration}.png')
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

# Main training function
def train(env, sb3_algo):
    # Generate a unique timestamp for this training run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{sb3_algo}_{timestamp}"
    
    # Set the path for saving the best model
    best_model_path = f"{model_dir}/best_{run_name}"
    os.makedirs(best_model_path, exist_ok=True)
    
    # Create a separate environment for model evaluation
    eval_env = RewardWrapper(gym.make('UnicycleBalance-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS))
    
    # Initialize the EvalCallback for saving the best model
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=best_model_path,
        log_path=log_dir, 
        eval_freq=10000,
        deterministic=True, 
        render=False,
    )

    # Initialize the UnicyclePositionLogger
    unicycle_logger = UnicyclePositionLogger(best_model_path)
    
    # Combine callbacks
    callbacks = [eval_callback, unicycle_logger]

    # Initialize the appropriate learning algorithm
    match sb3_algo:
        case 'SAC':
            model = SAC('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir)
        case 'TD3':
            model = TD3('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir)
        case 'A2C':
            model = A2C('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir)
        case _:
            print('Algorithm not found')
            return

    # Set the number of timesteps for each training iteration
    TIMESTEPS = 30000 
    iters = 0
    
    # Main training loop
    while True:
        iters += 1
        
        # Train the model
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=run_name, callback=callbacks)
        
        # Save the model
        model.save(f"{model_dir}/{run_name}_{TIMESTEPS*iters}")

        # Generate and save the unicycle position plot
        plot_unicycle_position(unicycle_logger, run_name, iters)

        # Add termination condition if needed

# Function to test the trained model
def test(env, sb3_algo, path_to_model):
    # Load the appropriate model based on the algorithm
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

    obs, _ = env.reset()
    terminated = truncated = False
    total_reward = 0
    step_count = 0
    
    # Run the episode
    while not (terminated or truncated) and step_count < MAX_EPISODE_STEPS:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        total_reward += reward
        step_count += 1
        
        unicycle_pos = obs[:3]
        quat_mujoco = obs[3:7]
        quat_scipy = np.roll(quat_mujoco, -1)  # Convert to SciPy quaternion order (xyzw)
        unicycle_euler = Rotation.from_quat(quat_scipy).as_euler('xyz', degrees=False)
        wheel_velocity = obs[14]  # Assuming wheel velocity is at index 14

        print(f"Step: {step_count}, Reward: {reward:.4f}")
        print(f"Unicycle Position: ({unicycle_pos[0]:.4f}, {unicycle_pos[1]:.4f}, {unicycle_pos[2]:.4f})")
        print(f"Unicycle Roll, Pitch: ({unicycle_euler[0]:.4f}, {unicycle_euler[1]:.4f})")
        print(f"Wheel Velocity: {wheel_velocity:.4f}")
        print("------------------------------")
    
    # Print episode summary
    print(f"\nEpisode finished after {step_count} steps")
    print(f"Total reward: {total_reward}")
    print(f"Average reward per step: {total_reward / step_count}")
    
    if info.get('goal_reached', False):
        print("Episode ended by reaching the goal!")
    elif terminated:
        print("Episode ended by termination condition.")
    elif truncated:
        print("Episode ended by truncation (max steps reached).")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    # Create and wrap the environment
    gymenv = RewardWrapper(gym.make('UnicycleBalance-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS))

    if args.train:
        train(gymenv, args.sb3_algo)

    if args.test:
        if os.path.isfile(args.test):
            # For testing, use render_mode='human'
            test_env = RewardWrapper(gym.make('UnicycleBalance-v0', render_mode='human', max_episode_steps=MAX_EPISODE_STEPS))
            test(test_env, args.sb3_algo, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')