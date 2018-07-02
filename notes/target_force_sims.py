import sys
import os
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot

sys.path.insert(0, os.path.abspath('..'))
from pymuscle import PotvinMuscleFibers
from pymuscle import PotvinMotorNeuronPool

motor_unit_count = 120
motor_unit_indices = np.arange(1, motor_unit_count + 1)

# Motor Neuron Pool
pool = PotvinMotorNeuronPool(motor_unit_count)

# Fibers
fibers = PotvinMuscleFibers(motor_unit_count)

# Target Force - 20% MVC
max_force = 2234.0
target_percent = 40
target_force = max_force * (target_percent / 100)
excitations = np.full(motor_unit_count, 1.0)
e_inc = 0.01
total_force = 0
sim_time = 0.0
sim_duration = 200.0
time_inc = 0.1
while sim_time < sim_duration:
    while total_force < target_force:
        firing_rates = pool.step(excitations)
        normalized_firing_rates = fibers._normalize_firing_rates(firing_rates)
        normalized_forcFaImes = fibers._calc_normalized_forces(normalized_firing_rates)
        inst_forces = fibers._calc_inst_forces(normalized_forces)
        total_force = fibers._calc_total_inst_force(inst_forces)
        excitations += e_inc
    sim_time += time_inc

print(excitations[0])
