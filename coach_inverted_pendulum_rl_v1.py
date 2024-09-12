import gymnasium as gym
from stable_baselines3 import SAC
import os
import argparse
import custom_gym.envs.mujoco
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import datetime
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Max episode steps and coach model path
MAX_EPISODE_STEPS = 3000
COACH_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pendulum_trained_model.zip")

class PendulumCoachLogger(BaseCallback):
    def __init__(self, best_model_path, verbose=0):
        super(PendulumCoachLogger, self).__init__(verbose)
        self.best_model_path = best_model_path
        self.rewards = []
        self.cart_positions = []
        self.student_actions_x = []
        self.student_actions_y = []
        self.coach_actions_x = []
        self.coach_actions_y = []
        self.combined_actions_x = []
        self.combined_actions_y = []
        self.timesteps = []
        self.episode_starts = [0]
        self.iteration_rewards = []

    def _on_step(self) -> bool:
        obs = self.locals['new_obs'][0]
        reward = self.locals['rewards'][0]
        student_action = self.locals['actions'][0]
        info = self.locals['infos'][0]
        coach_action = info.get('coach_action', np.zeros_like(student_action))

        self.rewards.append(reward)
        self.cart_positions.append(obs[:2])
        self.student_actions_x.append(student_action[0])
        self.student_actions_y.append(student_action[1])
        self.coach_actions_x.append(coach_action[0])
        self.coach_actions_y.append(coach_action[1])
        combined_action_x = student_action[0] + coach_action[0]
        combined_action_y = student_action[1] + coach_action[1]
        self.combined_actions_x.append(combined_action_x)
        self.combined_actions_y.append(combined_action_y)
        self.timesteps.append(self.num_timesteps)

        if self.locals['dones'][0]:
            self.episode_starts.append(self.num_timesteps)

        return True

    def on_rollout_end(self) -> None:
        if self.num_timesteps % 3000 == 0:
            self.iteration_rewards.append(self.rewards[-3000:])
            self.plot_graphs()

    def plot_graphs(self):
        iteration = self.num_timesteps // 3000
        fig, axs = plt.subplots(4, 2, figsize=(20, 25))

        # Reward graph (cumulative)
        cumulative_rewards = np.cumsum(self.rewards)
        axs[0, 0].plot(self.timesteps, cumulative_rewards)
        axs[0, 0].set_title('Cumulative Rewards over time')
        axs[0, 0].set_xlabel('Timesteps')
        axs[0, 0].set_ylabel('Cumulative Reward')

        # Cart position graph
        positions = np.array(self.cart_positions)
        axs[0, 1].plot(positions[:, 0], positions[:, 1])
        axs[0, 1].set_title('Cart Movement (Top-down view)')
        axs[0, 1].set_xlabel('X Position')
        axs[0, 1].set_ylabel('Y Position')
        axs[0, 1].axis('equal')

        # Student actions (X and Y separately)
        axs[1, 0].plot(self.timesteps, self.student_actions_x)
        axs[1, 0].set_title('Student Actions (X)')
        axs[1, 0].set_xlabel('Timesteps')
        axs[1, 0].set_ylabel('Action')

        axs[1, 1].plot(self.timesteps, self.student_actions_y)
        axs[1, 1].set_title('Student Actions (Y)')
        axs[1, 1].set_xlabel('Timesteps')
        axs[1, 1].set_ylabel('Action')

        # Coach actions (X and Y separately)
        axs[2, 0].plot(self.timesteps, self.coach_actions_x)
        axs[2, 0].set_title('Coach Actions (X)')
        axs[2, 0].set_xlabel('Timesteps')
        axs[2, 0].set_ylabel('Action')

        axs[2, 1].plot(self.timesteps, self.coach_actions_y)
        axs[2, 1].set_title('Coach Actions (Y)')
        axs[2, 1].set_xlabel('Timesteps')
        axs[2, 1].set_ylabel('Action')

        # Combined actions (X and Y separately)
        axs[3, 0].plot(self.timesteps, self.combined_actions_x)
        axs[3, 0].set_title('Combined Actions (X)')
        axs[3, 0].set_xlabel('Timesteps')
        axs[3, 0].set_ylabel('Action')
        axs[3, 0].axhline(y=0, color='r', linestyle='--')

        axs[3, 1].plot(self.timesteps, self.combined_actions_y)
        axs[3, 1].set_title('Combined Actions (Y)')
        axs[3, 1].set_xlabel('Timesteps')
        axs[3, 1].set_ylabel('Action')
        axs[3, 1].axhline(y=0, color='r', linestyle='--')

        plt.tight_layout()
        save_path = os.path.join(self.best_model_path, f'pendulum_coach_analysis_iteration_{iteration}.png')
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
        plt.close()

    def plot_final_reward_graph(self):
        plt.figure(figsize=(15, 10))
        iteration_rewards = [np.sum(self.rewards[i:i+3000]) for i in range(0, len(self.rewards), 3000)]
        plt.plot(range(1, len(iteration_rewards) + 1), iteration_rewards)
        plt.title('Total Rewards per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Total Reward')
        save_path = os.path.join(self.best_model_path, 'final_reward_graph.png')
        plt.savefig(save_path)
        print(f"Final reward graph saved to: {save_path}")
        plt.close()

def train(env, coach_model):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Pendulum_Coach_{timestamp}"

    best_model_path = f"{model_dir}/best_{run_name}"
    os.makedirs(best_model_path, exist_ok=True)

    eval_env = gym.make('InvertedPendulum3DWithCoach-v1', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)
    
    env.set_coach_model(coach_model)
    eval_env.set_coach_model(coach_model)
    print("Coach model set for training env:", env.coach_model is not None)
    print("Coach model set for eval env:", eval_env.coach_model is not None)

    print("Testing coach model:")
    test_obs = env.reset()[0]
    test_action, _ = coach_model.predict(test_obs, deterministic=True)
    print("Test observation:", test_obs)
    print("Test coach action:", test_action)
    env.reset()  # 환경 상태 리셋

    eval_callback = EvalCallback(eval_env, best_model_save_path=best_model_path,
                                 log_path=log_dir, eval_freq=10000,
                                 deterministic=True, render=False)

    pendulum_coach_logger = PendulumCoachLogger(best_model_path)
    
    callbacks = [eval_callback, pendulum_coach_logger]

    model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)

    TIMESTEPS = 3000
    total_timesteps = 0
    max_timesteps = 3000 * 100 # 3000 steps * 100 iterations

    while total_timesteps < max_timesteps:
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=run_name, callback=callbacks)
        model.save(f"{model_dir}/{run_name}_{total_timesteps}")
        
        total_timesteps += TIMESTEPS
        print(f"Total timesteps: {total_timesteps}")
        
        pendulum_coach_logger.plot_graphs()

    print("Training completed after reaching 300000 timesteps.")
    pendulum_coach_logger.plot_final_reward_graph()  # Final reward graph

def test(env, path_to_model, num_episodes=30):
    model = SAC.load(path_to_model, env=env)

    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        total_reward = 0
        step_count = 0

        print(f"\nEpisode {episode + 1}")

        while not (terminated or truncated) and step_count < MAX_EPISODE_STEPS:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            total_reward += reward
            step_count += 1

            if step_count % 1000 == 0:  # Print info every 1000 steps
                cart_x, cart_y = obs[0], obs[1]
                print(f"Step: {step_count}, Reward: {reward:.4f}, Angle: {info.get('angle', 'N/A'):.4f}, Position: ({cart_x:.4f}, {cart_y:.4f})")

        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)

        print(f"Episode {episode + 1} finished after {step_count} steps")
        print(f"Total reward: {total_reward}")
        print(f"Average reward per step: {total_reward / step_count}")

    print("\nTest Summary:")
    print(f"Average episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Number of episodes reaching max steps: {sum([1 for length in episode_lengths if length == MAX_EPISODE_STEPS])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model with coach.')
    parser.add_argument('-train', '--train', action='store_true')
    parser.add_argument('-test', '--test', metavar='path_to_model')
    args = parser.parse_args()

    # Load the coach model
    coach_model = SAC.load(COACH_MODEL_PATH)

    if args.train:
        gymenv = gym.make('InvertedPendulum3DWithCoach-v1', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)
        gymenv.set_coach_model(coach_model)
        train(gymenv, coach_model)

    if args.test:
        if os.path.isfile(args.test):
            gymenv = gym.make('InvertedPendulum3DWithCoach-v1', render_mode='human', max_episode_steps=MAX_EPISODE_STEPS)
            gymenv.set_coach_model(coach_model)
            test(gymenv, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')