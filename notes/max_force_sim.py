import sys
import os
import numpy as np
import plotly.graph_objs as go
import colorlover as cl
from plotly.offline import plot
from copy import copy

sys.path.insert(0, os.path.abspath('..'))
from pymuscle import PyMuscleFibers as Fibers
from pymuscle import PotvinFuglevand2017MotorNeuronPool as Pool

motor_unit_count = 120
motor_unit_indices = np.arange(1, motor_unit_count + 1)

# Motor Neuron Pool
pool = Pool(motor_unit_count)

# Fibers
fibers = Fibers(motor_unit_count)

# Setting colors for plot.
potvin_scheme = [
    'rgb(115, 0, 0)',
    'rgb(252, 33, 23)',
    'rgb(230, 185, 43)',
    'rgb(107, 211, 100)',
    'rgb(52, 211, 240)',
    'rgb(36, 81, 252)',
    'rgb(0, 6, 130)'
]
# It's hacky but also sorta cool.
c = cl.to_rgb(cl.interp(potvin_scheme, motor_unit_count))
c = [val.replace('rgb', 'rgba') for val in c]
c = [val.replace(')', ',{})') for val in c]


def get_color(trace_index: int) -> str:
    # The first and every 20th trace should be full opacity
    alpha = 0.2
    if trace_index == 0 or ((trace_index + 1) % 20 == 0):
        alpha = 1.0
    color = c[trace_index].format(alpha)
    return color

if False:
    all_firing_rates = []
    all_excitation_levels = []
    all_norm_forces = []
    all_norm_firing = []
    for i in np.arange(1.0, 67.0, 0.1):
        excitations = np.full(motor_unit_count, i)
        firing_rates = pool._calc_firing_rates(excitations)
        normalized_firing_rates = fibers._normalize_firing_rates(firing_rates)
        normalized_forces = fibers._calc_normalized_forces(normalized_firing_rates)
        current_forces = fibers._calc_current_forces(normalized_forces)
        total_force = sum(current_forces)
        all_excitation_levels.append(i)
        all_firing_rates.append(firing_rates[0])
        all_norm_forces.append(normalized_forces[0])
        all_norm_firing.append(normalized_firing_rates[0])

    plot([go.Scatter(
        x=all_excitation_levels,
        y=all_firing_rates,
        mode='markers'
    )], filename='firing-rates-by-excitation.html')

    plot([go.Scatter(
        x=all_firing_rates,
        y=all_norm_firing,
        mode='markers'
    )], filename='norm-firing-rates-by-firing-rate.html')

    plot([go.Scatter(
        x=all_norm_firing,
        y=all_norm_forces,
        mode='markers'
    )], filename='norm-forces-by-norm-firing.html')

    quit()


def get_force(excitations, cur_time):
    firing_rates = pool._calc_adapted_firing_rates(excitations, cur_time)
    normalized_firing_rates = fibers._normalize_firing_rates(firing_rates)
    normalized_forces = fibers._calc_normalized_forces(normalized_firing_rates)
    current_forces = fibers._calc_current_forces(normalized_forces)
    total_force = sum(current_forces)
    return firing_rates, normalized_forces, current_forces, total_force


# Target Force - 100% MVC
max_excitation = 67.0
e_inc = 0.01
sim_time = 0.0
max_force = 2216.0
sim_duration = 200.0
time_inc = 1.0
rest_start = 100
rest_end = 150
force_capacities = fibers._peak_twitch_forces
total_peak_capacity = sum(force_capacities)
step_counter = 0
all_forces = []
all_total_forces = []
all_excitation_levels = []
all_capacities = []
all_total_capacities = []
all_firing_rates = []
current_forces = []
hit_max_excite = False
excitations = np.full(motor_unit_count, max_excitation)
while sim_time < sim_duration:
    if sim_time > rest_start and sim_time < rest_end:
        excitations = np.full(motor_unit_count, 0)
    elif sim_time > rest_end:
        excitations = np.full(motor_unit_count, max_excitation)
    firing_rates, normalized_forces, current_forces, total_force = get_force(excitations, sim_time)
    # Record our step
    sim_time += time_inc
    # print("Sim time", sim_time)
    step_counter += 1
    all_forces.append(current_forces)
    all_total_forces.append((total_force / max_force) * 100)
    all_excitation_levels.append((excitations[0] / max_excitation) * 100)
    all_capacities.append((copy(fibers._current_peak_forces) / fibers._peak_twitch_forces) * 100)
    total_capacity = sum(fibers._current_peak_forces)
    all_total_capacities.append((total_capacity / max_force) * 100)
    all_firing_rates.append(firing_rates)
    # Update fatigue
    fibers._update_fatigue(normalized_forces, time_inc)

times = np.arange(0.0, sim_duration, time_inc)

if False:
    # 2.A
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
    # Per Motor Unit Force
    all_array = np.array(all_forces).T
    data = []
    for i, t in enumerate(all_array):
        trace = go.Scatter(
            x=times,
            y=t,
            name=i + 1,
            marker=dict(
                color=get_color(i)
            ),
        )
        data.append(trace)
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title='Motor Unit Forces by Time'
        ))
    plot(fig, filename='forces-by-time.html')

if False:
    # 2.B - Per Motor Unit Firing Rate by Time
    all_array = np.array(all_firing_rates).T
    data = []
    for i, t in enumerate(all_array):
        trace = go.Scatter(
            x=times,
            y=t,
            name=i + 1,
            marker=dict(
                color=get_color(i)
            ),
        )
        data.append(trace)
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title='Motor Unit Firing Rates by Time'
        ))
    plot(fig, filename='firing-rates-by-time.html')

if True:
    # 2.D - Per Motor Unit Force Capacities by Time
    all_array = np.array(all_capacities).T
    data = []
    for i, t in enumerate(all_array):
        trace = go.Scatter(
            x=times,
            y=t,
            name=i + 1,
            marker=dict(
                color=get_color(i)
            ),
        )
        data.append(trace)
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title='Motor Unit Force Capacities by Time at Max Excitation'
        ))
    plot(fig, filename='force-capacities-by-time-max.html')
