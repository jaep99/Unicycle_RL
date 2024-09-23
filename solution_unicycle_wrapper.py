import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np
import os

class SolutionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Finding home directory
        home_dir = os.path.expanduser("~")
        unicycle_rl_dir = self.find_unicycle_rl_dir(home_dir)
        if unicycle_rl_dir is None:
            raise FileNotFoundError("Unicycle_RL directory not found in home directory or its subdirectories")
        
        # Using solution model
        solution_model_path = os.path.join(unicycle_rl_dir, "solution_unicycle_pendulum_trajectory.zip")
        if not os.path.exists(solution_model_path):
            raise FileNotFoundError(f"Solution model file not found at {solution_model_path}")
        
        self.solution_model = SAC.load(solution_model_path)
        self.first_success = False

    def find_unicycle_rl_dir(self, start_path):
        for root, dirs, files in os.walk(start_path):
            if "Unicycle_RL" in dirs:
                return os.path.join(root, "Unicycle_RL")
        return None


    def step(self, action):
        # Action based on the student agent
        obs, student_reward, terminated, truncated, info = self.env.step(action)
        
        if not self.first_success:
            # Obtaining solution model's action
            solution_action, _ = self.solution_model.predict(obs, deterministic=True)
            
            # Action based on solution model
            new_obs, solution_reward, new_terminated, new_truncated, new_info = self.env.step(solution_action)
            
            # Ratio might need adjustment
            combined_reward = 0.7 * student_reward + 0.3 * solution_reward
            
            obs = new_obs
            terminated = terminated or new_terminated
            truncated = truncated or new_truncated
            info.update(new_info)
            info.update({
                'student_action': action,
                'solution_action': solution_action,
                'student_reward': student_reward,
                'solution_reward': solution_reward,
            })
        else:
            combined_reward = student_reward

        # Checking first success
        if info.get('goal_reached', False) and not self.first_success:
            self.first_success = True
            print("First success achieved! Now training without solution intervention.")

        return obs, combined_reward, terminated, truncated, info