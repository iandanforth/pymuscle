#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os

from os.path import dirname
path = dirname(os.path.abspath(__file__)) + "/assets/ballonstring.xml"
model = load_model_from_path(path)
sim = MjSim(model)
viewer = MjViewer(sim)

t = 0
total_steps = 100000
inc = 5.0 / total_steps
for i in range(1, total_steps + 1):
    sim.step()

    if i == 1:
        sim.model.tendon_lengthspring[0] = sim.model.tendon_lengthspring[0] / 2
    # sim.model.tendon_stiffness[0] = sim.model.tendon_stiffness[0] + 10.0
    print("Lengths", sim.data.ten_length)
    cur_length = sim.data.ten_length[0]
    print("Rest Length", sim.model.tendon_lengthspring)
    rest_length = sim.model.tendon_lengthspring[0]
    print("Stiffness", sim.model.tendon_stiffness)
    cur_stiffness = sim.model.tendon_stiffness[0]
    tension = 0
    # Hooke's Law
    if cur_length > rest_length:
        delta = cur_length - rest_length
        tension = delta * cur_stiffness

    print("Tension", tension)
    # print("Sensor 1", sim.data.sensordata[:3])
    # print("Sensor 1 mag", np.linalg.norm(sim.data.sensordata[:3]))
    # print("Sensor 2", sim.data.sensordata[3:])
    # print("Sensor 2 mag", np.linalg.norm(sim.data.sensordata[3:]))

    # sim.data.ctrl[0] = sim.data.ctrl[0] - 0.001
    # if i > 6000:
    #     sim.data.ctrl[0] = 0.0

    # print(sim.data.ctrl[0])
    # left_val = (math.sin(i / 400) * 2.5) - 2.5
    # sim.data.ctrl[0] = left_val
    # print(sim.data.qfrc_passive)
    # right_val = (math.sin(math.pi + i / 1000) * 2.5) - 2.5
    # sim.data.ctrl[1] = right_val
    viewer.render()
