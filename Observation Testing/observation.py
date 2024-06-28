import mujoco
import numpy as np

xml_path = "inverted_double_pendulum_3d.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

mujoco.mj_resetData(model, data)

def print_observation(step):
    print(f"Step: {step}")
    print("Position (qpos):", data.qpos)
    print("Velocity (qvel):", data.qvel)
    print("="*30)

print_observation(0)

for step in range(1, 50): 
    mujoco.mj_step(model, data)
    print_observation(step)
