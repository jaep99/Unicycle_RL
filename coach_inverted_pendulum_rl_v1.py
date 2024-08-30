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

class CoachStudentLogger(BaseCallback):
    def __init__(self, best_model_path, verbose=0):
        super(CoachStudentLogger, self).__init__(verbose)
        self.best_model_path = best_model_path
        self.pendulum_positions = []
        self.student_actions = []
        self.coach_actions = []
        self.rewards = []
        self.timesteps = []

    def _on_step(self) -> bool:
        obs = self.locals['new_obs'][0]
        self.pendulum_positions.append(obs[:2])
        self.student_actions.append(self.locals['actions'][0])
        self.coach_actions.append(self.locals['infos'][0]['coach_action'])
        self.rewards.append(self.locals['rewards'][0])
        self.timesteps.append(self.num_timesteps)
        return True

def plot_coach_student_data(logger, run_name, iteration):
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    # Pendulum XY position
    positions = np.array(logger.pendulum_positions)
    axs[0, 0].plot(positions[:, 0], positions[:, 1])
    axs[0, 0].set_title(f'Pendulum XY Position (Iteration {iteration})')
    axs[0, 0].set_xlabel('X Position')
    axs[0, 0].set_ylabel('Y Position')
    axs[0, 0].axis('equal')

    # Student actions
    student_actions = np.array(logger.student_actions)
    axs[0, 1].plot(logger.timesteps, student_actions)
    axs[0, 1].set_title('Student Actions')
    axs[0, 1].set_xlabel('Timesteps')
    axs[0, 1].set_ylabel('Action Value')
    axs[0, 1].legend([f'Action {i}' for i in range(student_actions.shape[1])])

    # Coach actions
    coach_actions = np.array(logger.coach_actions)
    axs[1, 0].plot(logger.timesteps, coach_actions)
    axs[1, 0].set_title('Coach Actions')
    axs[1, 0].set_xlabel('Timesteps')
    axs[1, 0].set_ylabel('Action Value')
    axs[1, 0].legend([f'Action {i}' for i in range(coach_actions.shape[1])])

    # Rewards
    axs[1, 1].plot(logger.timesteps, logger.rewards)
    axs[1, 1].set_title('Rewards')
    axs[1, 1].set_xlabel('Timesteps')
    axs[1, 1].set_ylabel('Reward')

    plt.tight_layout()

    save_path = os.path.join(logger.best_model_path, f'coach_student_analysis_iter_{iteration}.png')
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.close()

class CoachWrapper(gym.Wrapper):
    def __init__(self, env, coach_model_path):
        super().__init__(env)
        self.coach_model = SAC.load(coach_model_path)

    def step(self, action):
        coach_action, _ = self.coach_model.predict(self.observation, deterministic=True)
        delta_action = coach_action - action
        combined_action = action + delta_action

        obs, reward, terminated, truncated, info = self.env.step(combined_action)
        info['coach_action'] = coach_action
        self.observation = obs
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.observation, info = self.env.reset(**kwargs)
        return self.observation, info

def train(env, sb3_algo):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{sb3_algo}_{timestamp}"

    best_model_path = f"{model_dir}/best_{run_name}"
    os.makedirs(best_model_path, exist_ok=True)

    eval_env = CoachWrapper(gym.make('InvertedPendulum3DWithCoach-v1', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS), "inverted_pendulum_trained_model.zip")
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=best_model_path,
        log_path=log_dir, 
        eval_freq=10000,
        deterministic=True, 
        render=False,
    )

    coach_student_logger = CoachStudentLogger(best_model_path)

    callbacks = [eval_callback, coach_student_logger]

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

    TIMESTEPS = 30000 
    iters = 0

    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=run_name, callback=callbacks)
        model.save(f"{model_dir}/{run_name}_{TIMESTEPS*iters}")
        plot_coach_student_data(coach_student_logger, run_name, iters)

def test(env, sb3_algo, path_to_model):
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

    while not (terminated or truncated) and step_count < MAX_EPISODE_STEPS:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        total_reward += reward
        step_count += 1

        print(f"Step: {step_count}, Reward: {reward:.4f}")
        print(f"Pendulum Position: ({obs[0]:.4f}, {obs[1]:.4f})")
        print(f"Student Action: {action}")
        print(f"Coach Action: {info['coach_action']}")
        print("------------------------------")

    print(f"\nEpisode finished after {step_count} steps")
    print(f"Total reward: {total_reward}")
    print(f"Average reward per step: {total_reward / step_count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    gymenv = CoachWrapper(gym.make('InvertedPendulum3DWithCoach-v1', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS), "inverted_pendulum_trained_model.zip")

    if args.train:
        train(gymenv, args.sb3_algo)

    if args.test:
        if os.path.isfile(args.test):
            test_env = CoachWrapper(gym.make('InvertedPendulum3DWithCoach-v1', render_mode='human', max_episode_steps=MAX_EPISODE_STEPS), "inverted_pendulum_trained_model.zip")
            test(test_env, args.sb3_algo, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')