import sys
import os
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
from copy import copy

sys.path.insert(0, os.path.abspath('..'))
from pymuscle import PotvinMuscleFibers
from pymuscle import PotvinMotorNeuronPool

motor_unit_count = 120
motor_unit_indices = np.arange(1, motor_unit_count + 1)

# Motor Neuron Pool
pool = PotvinMotorNeuronPool(motor_unit_count)

# Fibers
fibers = PotvinMuscleFibers(motor_unit_count)


def get_force(excitations):
    firing_rates = pool._calc_firing_rates(excitations)
    normalized_firing_rates = fibers._normalize_firing_rates(firing_rates)
    normalized_forces = fibers._calc_normalized_forces(normalized_firing_rates)
    current_forces = fibers._calc_current_forces(normalized_forces)
    total_force = fibers._calc_total_inst_force(current_forces)
    return firing_rates, normalized_forces, current_forces, total_force


# Target Force - 20% MVC
max_force = 2216.0
max_excitation = 67.0
target_percent = 20
target_force = max_force * (target_percent / 100)
e_inc = 0.01
hit_max_excite = False
# Find the required excitation level for target_force

starting_excitations = []
for i in range(1, 101):
    total_force = 0.0
    target_force = (i / 100) * max_force
    excitations = np.full(motor_unit_count, 1.0)
    while total_force < target_force:
        if excitations[0] > max_excitation:
            if not hit_max_excite:
                print("Hit max excitation")
                hit_max_excite = True
            break
        _, _, _, total_force = get_force(excitations)
        excitations += e_inc

    starting_excitations.append(int(round((excitations[0] - e_inc) * 100)))

print(starting_excitations)
