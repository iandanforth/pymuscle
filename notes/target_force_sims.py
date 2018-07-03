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


def get_force(excitations):
    firing_rates = pool.step(excitations)
    normalized_firing_rates = fibers._normalize_firing_rates(firing_rates)
    normalized_forces = fibers._calc_normalized_forces(normalized_firing_rates)
    current_forces = fibers._calc_current_forces(normalized_forces)
    total_force = fibers._calc_total_inst_force(current_forces)
    return normalized_forces, current_forces, total_force


# Target Force - 20% MVC
max_force = 2216.0
max_excitation = 67.0
target_percent = 20
target_force = max_force * (target_percent / 100)
e_inc = 0.01
sim_time = 0.0
sim_duration = 550.0
time_inc = 0.1
force_capacities = fibers._peak_twitch_forces
total_peak_capacity = sum(force_capacities)
step_counter = 0
all_forces = []
all_total_forces = []
all_excitation_levels = []
all_total_capacities = []
current_forces = []
hit_max_excite = False
excitations = np.full(motor_unit_count, 1.0)
while sim_time < sim_duration:
    total_force = 0.0
    excitations = excitations - (20 * e_inc) # Start the search again from a slightly lower value
    # Find the required excitation level for target_force
    while total_force < target_force:
        # print(excitations[0])
        if excitations[0] > max_excitation:
            if not hit_max_excite:
                print("Hit max excitation: ", sim_time)
                hit_max_excite = True
            break
        _, _, total_force = get_force(excitations)
        excitations += e_inc

    # We're now at the correct excitation level to generate target_force
    normalized_forces, current_forces, total_force = get_force(excitations)
    # Update fatigue
    inst_fatigue = fibers._calc_inst_fatigues(normalized_forces)
    fibers._apply_fatigue(inst_fatigue, time_inc)
    # Record our step
    sim_time += time_inc
    # print("Sim time", sim_time)
    step_counter += 1
    all_forces.append(current_forces)
    all_total_forces.append(total_force / max_force)
    all_excitation_levels.append(excitations[0] / max_excitation)
    total_capacity = sum(fibers._current_twitch_forces)
    all_total_capacities.append(total_capacity / total_peak_capacity)

# Per Motor Unit Force
all_array = np.array(all_forces).T
data = []
times = np.arange(0.0, sim_duration, time_inc)

if True:
    a_data = []
    # Total Muscle Force
    total_force_trace = go.Scatter(
        x=times,
        y=all_total_forces,
        name='Muscle force'
    )
    a_data.append(total_force_trace)
    # Total Excitation
    excitation_trace = go.Scatter(
        x=times,
        y=all_excitation_levels,
        name='Excitation'
    )
    a_data.append(excitation_trace)
    # Force Capacity
    capacity_trace = go.Scatter(
        x=times,
        y=all_total_capacities,
        name='Force Capacity'
    )
    a_data.append(capacity_trace)
    fig = go.Figure(
        data=a_data,
        layout=go.Layout(
            title='Motor Unit Forces by Time'
        )
    )
    plot(fig, filename='fig-2-a.html')

if False:
    for i, t in enumerate(all_array):
        trace = go.Scatter(
            x=times,
            y=t,
            name=i + 1
        )
        data.append(trace)
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title='Motor Unit Forces by Time'
        ))
    plot(fig, filename='forces-by-time.html')
