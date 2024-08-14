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