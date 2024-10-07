import gymnasium as gym

class NoSolutionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.success_count = 0
        self.total_steps = 0
        self.steps_since_last_success = 0

    def reset(self, **kwargs):
        self.steps_since_last_success = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.total_steps += 1
        self.steps_since_last_success += 1

        info.update({
            'student_action': action,
            'solution_action': [0, 0, 0],
            'total_steps': self.total_steps,
            'steps_since_last_success': self.steps_since_last_success,
            'success_count': self.success_count
        })

        if info.get('goal_reached', False):
            self.success_count += 1
            self.steps_since_last_success = 0
            
            if self.success_count % 1000 == 0:
                print(f"Goal reached! Success count: {self.success_count}, Total steps: {self.total_steps}")

        return obs, reward, terminated, truncated, info