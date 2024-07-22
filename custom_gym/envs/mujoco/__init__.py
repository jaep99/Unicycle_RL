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