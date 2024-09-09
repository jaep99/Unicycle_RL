import gymnasium as gym
from stable_baselines3 import SAC
import os
import argparse
import custom_gym.envs.mujoco
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import datetime
import numpy as np
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
        self.student_actions = []
        self.coach_actions = []
        self.timesteps = []
        self.episode_starts = [0]

    def _on_step(self) -> bool:
        obs = self.locals['new_obs'][0]
        reward = self.locals['rewards'][0]
        student_action = self.locals['actions'][0]
        coach_action = self.training_env.get_attr('coach_action')[0]

        self.rewards.append(reward)
        self.cart_positions.append(obs[:2])  # x and y positions
        self.student_actions.append(student_action)
        self.coach_actions.append(coach_action)
        self.timesteps.append(self.num_timesteps)

        if self.locals['dones'][0]:
            self.episode_starts.append(self.num_timesteps)

        return True

    def on_rollout_end(self) -> None:
        if self.num_timesteps % 3000 == 0:  # Every 3000 steps
            self.plot_graphs()

    def plot_graphs(self):
        iteration = self.num_timesteps // 3000
        fig, axs = plt.subplots(3, 2, figsize=(20, 20))

        # Reward graph
        axs[0, 0].plot(self.timesteps, self.rewards)
        axs[0, 0].set_title('Rewards over time')
        axs[0, 0].set_xlabel('Timesteps')
        axs[0, 0].set_ylabel('Reward')

        # Cart position graph
        positions = np.array(self.cart_positions)
        axs[0, 1].plot(positions[:, 0], positions[:, 1])
        axs[0, 1].set_title('Cart Movement (Top-down view)')
        axs[0, 1].set_xlabel('X Position')
        axs[0, 1].set_ylabel('Y Position')
        axs[0, 1].axis('equal')

        # Student actions
        axs[1, 0].plot(self.timesteps, self.student_actions)
        axs[1, 0].set_title('Student Actions')
        axs[1, 0].set_xlabel('Timesteps')
        axs[1, 0].set_ylabel('Action')

        # Coach actions
        axs[1, 1].plot(self.timesteps, self.coach_actions)
        axs[1, 1].set_title('Coach Actions')
        axs[1, 1].set_xlabel('Timesteps')
        axs[1, 1].set_ylabel('Action')

        # Average reward per episode
        episode_rewards = np.split(np.array(self.rewards), self.episode_starts[1:])
        avg_rewards = [np.mean(ep) for ep in episode_rewards if len(ep) > 0]
        axs[2, 0].plot(range(1, len(avg_rewards) + 1), avg_rewards)
        axs[2, 0].set_title('Average Reward per Episode')
        axs[2, 0].set_xlabel('Episode')
        axs[2, 0].set_ylabel('Average Reward')

        # Difference of Student and Coach action
        action_diff = np.array(self.student_actions) - np.array(self.coach_actions)
        axs[2, 1].plot(self.timesteps, action_diff)
        axs[2, 1].set_title('Action Difference (Student - Coach)')
        axs[2, 1].set_xlabel('Timesteps')
        axs[2, 1].set_ylabel('Action Difference')

        plt.tight_layout()
        save_path = os.path.join(self.best_model_path, f'pendulum_coach_analysis_iteration_{iteration}.png')
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
        plt.close()

def train(env, coach_model):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Pendulum_Coach_{timestamp}"

    best_model_path = f"{model_dir}/best_{run_name}"
    os.makedirs(best_model_path, exist_ok=True)

    eval_env = gym.make('InvertedPendulum3DWithCoach-v1', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)
    eval_env.set_coach_model(coach_model)
    
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