from gymnasium.envs.registration import register

register(
    id='InvertedPendulum3D-v0',
    entry_point='custom_gym.envs.mujoco.inverted_pendulum_3d_v0:InvertedPendulum3DEnv',
)

register(
    id='InvertedPendulum3D-v1',
    entry_point='custom_gym.envs.mujoco.inverted_pendulum_3d_v1:InvertedPendulum3DEnv',
)

register(
    id='InvertedPendulum3D-v2',
    entry_point='custom_gym.envs.mujoco.inverted_pendulum_3d_v2:InvertedPendulum3DEnv',
)

register(
    id='InvertedPendulum3D-v3',
    entry_point='custom_gym.envs.mujoco.coach_inverted_pendulum_3d_v0:InvertedPendulum3DEnv',
    max_episode_steps=30000,
)

register(
    id='InvertedPendulum3DWithCoach-v0',
    entry_point='custom_gym.envs.mujoco.coach_inverted_pendulum_3d_v0:InvertedPendulum3DEnvWithCoach',
    kwargs={'render_mode': None}, 
    max_episode_steps=30000,
)

register(
    id='InvertedPendulum3DWithCoach-v1',
    entry_point='custom_gym.envs.mujoco.coach_inverted_pendulum_3d_v1:InvertedPendulum3DWithCoach',
)

register(
    id='InvertedDoublePendulum3D-v0',
    entry_point='custom_gym.envs.mujoco.inverted_double_pendulum_3d_v0:InvertedDoublePendulum3DEnv',
)

register(
    id='PendulumTrajectory-v0',
    entry_point='custom_gym.envs.mujoco.pendulum_trajectory_v0:PendulumTrajectory',
)

register(
    id='UnicycleBalance-v0',
    entry_point='custom_gym.envs.mujoco.unicycle_balance_v0:UnicycleBalance',
)

register(
    id='UnicycleTrajectory-v0',
    entry_point='custom_gym.envs.mujoco.unicycle_trajectory_v0:UnicycleTrajectory',
)

register(
    id='UnicycleTurningTrajectory-v0',
    entry_point='custom_gym.envs.mujoco.unicycle_turning_trajectory_v0:UnicycleTurningTrajectory',
)

register(
    id='UnicyclePendulumBalance-v0',
    entry_point='custom_gym.envs.mujoco.unicycle_pendulum_balance_3d_v0:UnicyclePendulumBalance',
)

register(
    id='UnicyclePendulumTrajectory-v0',
    entry_point='custom_gym.envs.mujoco.unicycle_pendulum_trajectory_3d_v0:UnicyclePendulumTrajectory',
)

register(
    id='UnicyclePendulumTrajectoryCurriculum-v0',
    entry_point='custom_gym.envs.mujoco.unicycle_pendulum_trajectory_curriculum_3d_v0:UnicyclePendulumTrajectoryCurriculum',
)

