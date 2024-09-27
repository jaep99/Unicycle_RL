import gymnasium as gym

class NoSolutionWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.success_count = 0

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        info.update({
            'student_action': action,
            'solution_action': [0, 0, 0],
        })

        if info.get('goal_reached', False):
            self.success_count += 1
            print(f"Goal reached! Success count: {self.success_count}")

        return obs, reward, terminated, truncated, info