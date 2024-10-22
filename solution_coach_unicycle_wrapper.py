import gymnasium as gym
from gym.spaces import Box
from stable_baselines3 import SAC, PPO
import os
import numpy as np
from collections import deque

class SolutionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        home_dir = os.path.expanduser("~")
        unicycle_rl_dir = self.find_unicycle_rl_dir(home_dir)
        if unicycle_rl_dir is None:
            raise FileNotFoundError("Unicycle_RL directory not found in home directory or its subdirectories")
        
        solution_model_path = os.path.join(unicycle_rl_dir, "solution_unicycle_pendulum_trajectory.zip")
        if not os.path.exists(solution_model_path):
            raise FileNotFoundError(f"Solution model file not found at {solution_model_path}")
        
        self.solution_model = SAC.load(solution_model_path)
        self.coach_model = PPO("MlpPolicy", env, verbose=1)
        self.coach_model.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.cumulative_reward = deque(maxlen=10)
        self.prev_cumulative_reward = 0
        self.success_count = 0

        self.first_success = False

    def find_unicycle_rl_dir(self, start_path):
        for root, dirs, files in os.walk(start_path):
            if "Unicycle_RL" in dirs:
                return os.path.join(root, "Unicycle_RL")
        return None

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        if not self.first_success:
            obs, student_reward, terminated, truncated, info = self.env.step(action)

            # Coach model's weight prediction
            student_weight = self.coach_model.predict(obs)[0]
            solution_weight = 1 - student_weight

            solution_action, _ = self.solution_model.predict(obs, deterministic=True)
            new_obs, solution_reward, new_terminated, new_truncated, new_info = self.env.step(solution_action)
            
            combined_reward = student_weight * student_reward + solution_weight * solution_reward
            
            self.cumulative_reward += student_reward
            obs = new_obs
            terminated = terminated or new_terminated
            truncated = truncated or new_truncated
            info.update(new_info)

            # Progress reward
            progress_reward = self.calculate_progress_reward()
            combined_reward += progress_reward

            # Update the coach model
            self.coach_model.learn(total_timesteps = 1)
        else:
            obs, combined_reward, terminated, truncated, info = self.env.step(action)
            solution_action = [0, 0, 0]  # No action from solution model

        info.update({
            'student_action': action,
            'solution_action': solution_action,
            'student_weight': student_weight,
            'solution_weight': solution_weight
        })

        if info.get('goal_reached', False) and not self.first_success:
            self.first_success = True
            print("First success achieved! Now training without solution intervention.")

        return obs, combined_reward, terminated, truncated, info
    
    def calculate_progress_reward(self):
        if len(self.cumulative_reward) < 2:
            return 0
        
        recent_avg_reward = np.mean(list(self.cumulative_reward)[:-1])
        progress_reward = 0.1 * (self.cumulative_reward[-1] - recent_avg_reward)

        return progress_reward