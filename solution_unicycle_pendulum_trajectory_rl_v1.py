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
from solution_unicycle_wrapper import SolutionWrapper

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Maximum number of steps per episode
MAX_EPISODE_STEPS = 10000

class UnicyclePositionLogger(BaseCallback):
    def __init__(self, best_model_path, verbose=0):
        super(UnicyclePositionLogger, self).__init__(verbose)
        self.best_model_path = best_model_path
        self.unicycle_positions = []
        self.student_actions = []
        self.solution_actions = []
        self.timesteps = []
        self.success_timesteps = []
        self.success_counts = []
        self.total_successes = 0

    def _on_step(self) -> bool:
        obs = self.locals['new_obs'][0]
        info = self.locals['infos'][0]
        self.unicycle_positions.append(obs[:3])
        
        self.student_actions.append(info.get('student_action', [0, 0, 0]))
        self.solution_actions.append(info.get('solution_action', [0, 0, 0]))
        
        self.timesteps.append(self.num_timesteps)
        
        if info.get('goal_reached', False):
            self.total_successes += 1
            self.success_timesteps.append(self.num_timesteps)
            self.success_counts.append(self.total_successes)
        
        return True

def plot_unicycle_position(logger, run_name, iteration):
    fig = plt.figure(figsize=(20, 20))
    
    # 3D plot
    ax = fig.add_subplot(321, projection='3d')
    positions = np.array(logger.unicycle_positions)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.set_title(f'Unicycle 3D Movement (Iteration {iteration})')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_xlim(0, 12)  # Set x-axis limit to 0-12 meters
    
    # Top-down view
    ax = fig.add_subplot(322)
    ax.plot(positions[:, 0], positions[:, 1])
    ax.set_title(f'Unicycle Movement (Top-down view, Iteration {iteration})')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_xlim(0, 12)  # Set x-axis limit to 0-12 meters
    ax.axis('equal')
    
    # Student's actions
    ax = fig.add_subplot(323)
    student_actions = np.array(logger.student_actions)
    ax.plot(logger.timesteps, student_actions)
    ax.set_title('Student Actions')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Action Values')
    ax.legend(['Action 1', 'Action 2', 'Action 3'])

    # Solution's actions
    ax = fig.add_subplot(324)
    solution_actions = np.array(logger.solution_actions)
    ax.plot(logger.timesteps, solution_actions)
    ax.set_title('Solution Actions')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Action Values')
    ax.legend(['Action 1', 'Action 2', 'Action 3'])

    # Success count
    ax = fig.add_subplot(325)
    ax.scatter(logger.success_timesteps, logger.success_counts, color='red', marker='o')
    ax.set_title('Cumulative Success Count')
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Number of Successes')
    ax.grid(True)

    # Ensure all plots have the same x-axis range for timestep-based plots
    max_timestep = max(logger.timesteps)
    for subplot in [fig.axes[2], fig.axes[3], fig.axes[4]]:  # Only adjust timestep-based plots
        subplot.set_xlim([0, max_timestep])

    plt.tight_layout()
    
    save_path = os.path.join(logger.best_model_path, f'unicycle_analysis_iter_{iteration}.png')
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

def train(env):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"SAC_{timestamp}"
    
    best_model_path = f"{model_dir}/best_{run_name}"
    os.makedirs(best_model_path, exist_ok=True)
    
    eval_env = SolutionWrapper(gym.make('SolutionUnicyclePendulumTrajectory-v1', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS))
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=best_model_path,
        log_path=log_dir, 
        eval_freq=10000,
        deterministic=True, 
        render=False,
    )

    unicycle_logger = UnicyclePositionLogger(best_model_path)
    
    callbacks = [eval_callback, unicycle_logger]

    model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)

    total_timesteps = 0
    start_time = time.time()
    prev_success_count = 0

    while True:
        model.learn(total_timesteps=10000, reset_num_timesteps=False, tb_log_name=run_name, callback=callbacks)
        total_timesteps += 10000
        
        current_success_count = env.env.success_count  # Access the unwrapped environment
        new_successes = current_success_count - prev_success_count
        
        print(f"Total timesteps: {total_timesteps}, Total Successes: {current_success_count}, New Successes: {new_successes}")
        
        if new_successes > 0:
            print(f"Goal reached! New success count: {new_successes}")
        
        prev_success_count = current_success_count

        if current_success_count >= 10:
            end_time = time.time()
            training_time = end_time - start_time
            print(f"\nTraining completed! 10 successes achieved.")
            print(f"Total training time: {training_time:.2f} seconds")
            print(f"Total timesteps: {total_timesteps}")
            break

    model.save(f"{model_dir}/{run_name}_final")

    plot_unicycle_position(unicycle_logger, run_name, total_timesteps // 10000)

def test(env, path_to_model):
    model = SAC.load(path_to_model, env=env)

    obs, _ = env.reset()
    terminated = truncated = False
    total_reward = 0
    step_count = 0
    success_count = 0
    
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
    parser = argparse.ArgumentParser(description='Train or test SAC model.')
    parser.add_argument('-train', '--train', action='store_true')
    parser.add_argument('-test', '--test', metavar='path_to_model')
    args = parser.parse_args()

    gymenv = SolutionWrapper(gym.make('SolutionUnicyclePendulumTrajectory-v1', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS))

    if args.train:
        train(gymenv)

    if args.test:
        if os.path.isfile(args.test):
            test_env = SolutionWrapper(gym.make('SolutionUnicyclePendulumTrajectory-v1', render_mode='human', max_episode_steps=MAX_EPISODE_STEPS))
            test(test_env, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')