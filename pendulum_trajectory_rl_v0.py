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

# Maximum number of steps per episode (30000 is long enough to train)
MAX_EPISODE_STEPS = 30000

# Custom callback to log cart positions
class CartPositionLogger(BaseCallback):
    def __init__(self, best_model_path, verbose=0):
        super(CartPositionLogger, self).__init__(verbose)
        self.best_model_path = best_model_path
        self.cart_positions_x = []
        self.cart_positions_y = []
        self.timesteps = []

    def _on_step(self) -> bool:
        # Extract x and y positions from the observation
        x, y = self.locals['new_obs'][0][:2]
        self.cart_positions_x.append(x)
        self.cart_positions_y.append(y)
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

# Function to plot cart positions
def plot_cart_position(logger, run_name, iteration):
    plt.figure(figsize=(15, 15))
    
    # X vs Time plot
    plt.subplot(2, 2, 1)
    plt.plot(logger.timesteps, logger.cart_positions_x)
    plt.title(f'Cart X Position vs Time (Iteration {iteration})')
    plt.xlabel('Timesteps')
    plt.ylabel('X Position')
    
    # Y vs Time plot
    plt.subplot(2, 2, 2)
    plt.plot(logger.timesteps, logger.cart_positions_y)
    plt.title(f'Cart Y Position vs Time (Iteration {iteration})')
    plt.xlabel('Timesteps')
    plt.ylabel('Y Position')
    
    # X vs Y plot (Top-down view of cart movement)
    plt.subplot(2, 2, (3, 4))
    plt.plot(logger.cart_positions_x, logger.cart_positions_y)
    plt.title(f'Cart Movement (Top-down view, Iteration {iteration})')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')  # This ensures the scale is the same for both axes
    
    # Add color gradient to show direction of movement
    points = plt.scatter(logger.cart_positions_x, logger.cart_positions_y, 
                         c=logger.timesteps, cmap='viridis', s=1)
    plt.colorbar(points, label='Timestep')
    
    # Mark start and end points
    plt.scatter(logger.cart_positions_x[0], logger.cart_positions_y[0], color='red', s=100, label='Start')
    plt.scatter(logger.cart_positions_x[-1], logger.cart_positions_y[-1], color='green', s=100, label='End')
    plt.legend()

    plt.tight_layout()
    
    # Save the plot in the best model directory
    os.makedirs(logger.best_model_path, exist_ok=True)
    save_path = os.path.join(logger.best_model_path, f'cart_position_iter_{iteration}.png')
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")  # Print the save location
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
    eval_env = RewardWrapper(gym.make('PendulumTrajectory-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS))
    
    # Initialize the EvalCallback for saving the best model
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=best_model_path,
        log_path=log_dir, 
        eval_freq=10000,
        deterministic=True, 
        render=False,
    )

    # Initialize the CartPositionLogger
    cart_logger = CartPositionLogger(best_model_path)
    
    # Combine callbacks
    callbacks = [eval_callback, cart_logger]

    # Initialize the appropriate learning algorithm
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

        # Generate and save the cart position plot
        plot_cart_position(cart_logger, run_name, iters)

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
        
        # Extract cart position and pole angles
        cart_x, cart_y = obs[0], obs[1]
        quat1, quat2 = obs[2:6], obs[6:10]
        
        # Convert quaternions to Euler angles
        r1 = Rotation.from_quat(quat1)
        r2 = Rotation.from_quat(quat2)
        euler1 = r1.as_euler('xyz', degrees=True)
        euler2 = r2.as_euler('xyz', degrees=True)

        # Print step information
        print(f"Step: {step_count}, Reward: {reward:.4f}")
        print(f"Cart Position: ({cart_x:.4f}, {cart_y:.4f})")
        print(f"Pole 1 Angles (degrees): ({euler1[0]:.4f}, {euler1[1]:.4f}, {euler1[2]:.4f})")
        print(f"Pole 2 Angles (degrees): ({euler2[0]:.4f}, {euler2[1]:.4f}, {euler2[2]:.4f})")
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
    gymenv = RewardWrapper(gym.make('PendulumTrajectory-v0', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS))

    if args.train:
        train(gymenv, args.sb3_algo)

    if args.test:
        if os.path.isfile(args.test):
            # For testing, use render_mode='human'
            test_env = RewardWrapper(gym.make('PendulumTrajectory-v0', render_mode='human', max_episode_steps=MAX_EPISODE_STEPS))
            test(test_env, args.sb3_algo, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')