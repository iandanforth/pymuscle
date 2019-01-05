#!/usr/bin/env python3
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os

from os.path import dirname
path = dirname(os.path.abspath(__file__)) + "/assets/minimal-gen.xml"
model = load_model_from_path(path)
sim = MjSim(model)
viewer = MjViewer(sim)

t = 0
total_steps = 15000
inc = 5.0 / total_steps
time_step = 0.002
forces = []
sensor_forces = []
lengths = []
act_forces = []
initial_stiffness = None
total_forces = []
prev_length = None
for i in range(1, total_steps + 1):
    sim.step()
    viewer.render()
