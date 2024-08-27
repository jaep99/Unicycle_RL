import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C
import os
import argparse
import custom_gym.envs.mujoco
from stable_baselines3.common.callbacks import EvalCallback
import datetime

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Max episode steps added
MAX_EPISODE_STEPS = 30000

def train(env, sb3_algo):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{sb3_algo}_{timestamp}"

    # Create a coach agent
    coach_agent = CoachAgent(env.action_space)

    # Wrap the environment with the coach
    wrapped_env = InvertedPendulum3DEnvWithCoach(env, coach_agent)

    # Create seperated environment for model evaluation added
    #eval_env = gym.make('InvertedPendulum3D-v1', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)
    eval_env = InvertedPendulum3DEnvWithCoach(
        gym.make('InvertedPendulum3D-v1', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS),
        coach_agent
    )

    # EvalCallback added
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"{model_dir}/best_{run_name}",
                                 log_path=log_dir, eval_freq=10000,
                                 deterministic=True, render=False)

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

    TIMESTEPS = 100000 
    iters = 0
    while True:
        iters += 1
        
        # eval_callback added
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=run_name, callback=eval_callback)
        model.save(f"{model_dir}/{run_name}_{TIMESTEPS*iters}")

def test(env, sb3_algo, path_to_model):
    coach_agent = CoachAgent(env.action_space)
    wrapped_env = InvertedPendulum3DEnvWithCoach(env, coach_agent)

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

    obs, _ = wrapped_env.reset()
    terminated = truncated = False
    total_reward = 0
    step_count = 0
    
    while not (terminated or truncated) and step_count < MAX_EPISODE_STEPS:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        wrapped_env.render()
        
        total_reward += reward
        step_count += 1
        
        cart_x, cart_y = obs[0], obs[1]

        # Print step information
        print(f"Step: {step_count}, Reward: {reward:.4f}, Angle: {info.get('angle', 'N/A'):.4f}, Position: ({cart_x:.4f}, {cart_y:.4f})")
        
    print(f"\nEpisode finished after {step_count} steps")
    print(f"Total reward: {total_reward}")
    print(f"Average reward per step: {total_reward / step_count}")

if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    gymenv = gym.make('InvertedPendulum3D-v1', render_mode=None, max_episode_steps=MAX_EPISODE_STEPS)

    if args.train:
        train(gymenv, args.sb3_algo)

    if args.test:
        if os.path.isfile(args.test):
            gymenv = gym.make('InvertedPendulum3D-v1', render_mode='human', max_episode_steps=MAX_EPISODE_STEPS)
            test(gymenv, args.sb3_algo, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')
