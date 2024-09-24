import gymnasium as gym
from stable_baselines3 import SAC
import os
import numpy as np

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
        self.first_success = False
        self.episode_first_success = False

    def find_unicycle_rl_dir(self, start_path):
        for root, dirs, files in os.walk(start_path):
            if "Unicycle_RL" in dirs:
                return os.path.join(root, "Unicycle_RL")
        return None

    def reset(self, **kwargs):
        self.episode_first_success = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, student_reward, terminated, truncated, info = self.env.step(action)
        
        if not self.episode_first_success:
            solution_action, _ = self.solution_model.predict(obs, deterministic=True)
            new_obs, solution_reward, new_terminated, new_truncated, new_info = self.env.step(solution_action)
            
            combined_reward = 0.7 * student_reward + 0.3 * solution_reward
            
            obs = new_obs
            terminated = terminated or new_terminated
            truncated = truncated or new_truncated
            info.update(new_info)
        else:
            combined_reward = student_reward
            solution_action = [0, 0, 0]  # No action from solution model

        info.update({
            'student_action': action,
            'solution_action': solution_action,
            'student_reward': student_reward,
            'solution_reward': solution_reward if not self.episode_first_success else 0,
        })

        if info.get('goal_reached', False) and not self.first_success:
            self.first_success = True
            self.episode_first_success = True
            print("First success achieved! Now training without solution intervention.")
        
        if info.get('goal_reached', False):
            self.episode_first_success = True

        return obs, combined_reward, terminated, truncated, info