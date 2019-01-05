import sys
import os
import numpy as np
import plotly.graph_objs as go
import colorlover as cl
from plotly.offline import plot
from copy import copy

sys.path.insert(0, os.path.abspath('..'))
from pymuscle import PotvinFuglevand2017MuscleFibers as Fibers
from pymuscle import PotvinFuglevand2017MotorNeuronPool as Pool

motor_unit_count = 120
motor_unit_indices = np.arange(1, motor_unit_count + 1)

# Motor Neuron Pool
apply_fatigue = True
pool = Pool(motor_unit_count, apply_fatigue=apply_fatigue)

# Fibers
fibers = Fibers(motor_unit_count)  # Disable fatigue below if desired


def get_force(excitations, step_size):
    firing_rates = pool.step(excitations, step_size)
    normalized_firing_rates = fibers._normalize_firing_rates(firing_rates)
    normalized_forces = fibers._calc_normalized_forces(normalized_firing_rates)
    current_forces = fibers._calc_current_forces(normalized_forces)
    total_force = np.sum(current_forces)
    return firing_rates, normalized_forces, current_forces, total_force


# Target Force
max_force = 2216.0
max_excitation = 67.0
target_percent = 100
target_force = max_force * (target_percent / 100)
e_inc = 0.01
sim_time = 0.0
sim_duration = 40.0
time_inc = 1.0
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
excitations = np.full(motor_unit_count, 1.0)
while sim_time < sim_duration:
    total_force = 0.0
    excitations = excitations - (2 * e_inc)  # Start the search again from a slightly lower value
    # Find the required excitation level for target_force
    while total_force < target_force:
        if excitations[0] > max_excitation:
            if not hit_max_excite:
                print("Hit max excitation: ", sim_time)
                hit_max_excite = True
            break
        _, _, _, total_force = get_force(excitations, 0) # Don't advance time here
        excitations += e_inc

    # We're now at the correct excitation level to generate target_force
    firing_rates, normalized_forces, current_forces, total_force = get_force(excitations, time_inc)
    # Update fatigue
    if apply_fatigue:
        fibers._update_fatigue(normalized_forces, time_inc)
    # Record our step
    sim_time += time_inc
    step_counter += 1
    all_forces.append(current_forces)
    all_total_forces.append(total_force / max_force)
    all_excitation_levels.append(excitations[0] / max_excitation)
    all_capacities.append(copy(fibers._current_peak_forces) / fibers._peak_twitch_forces)
    total_capacity = sum(fibers._current_peak_forces)
    all_total_capacities.append(total_capacity / total_peak_capacity)
    all_firing_rates.append(firing_rates)


###############################################################################
# Visualization Code

times = np.arange(0.0, sim_duration, time_inc)

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


def get_annotation(
    trace_index: int,
    x: int,
    y_offset: float,
    trace_data: list,
    step_size: float
) -> dict:
    """
    Line annotations.
    """
    y = trace_data[int(round(x / step_size))] + y_offset
    trace_index += 1
    prefix = ""
    if trace_index == 1:
        prefix = "MU"
    text = "<em>{}{}</em>".format(prefix, str(trace_index))
    annotation = dict(
        x=x,
        y=y,
        text=text,
        font=dict(
            family='Arial',
            size=16,
            color=get_color(i),
        ),
        showarrow=False
    )
    return annotation


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
            title='Excitation, Force, & Force capacity by Time',
            yaxis=dict(
                title='Excitation, force, & force capacity (% max.)'
            ),
            xaxis=dict(
                title='Time (s)'
            )
        )
    )
    plot(fig, filename='totals-by-time.html')

if False:
    # 2.C
    # Per Motor Unit Force
    all_array = np.array(all_forces).T
    data = []
    annotations = []
    anno_offsets = {
        0: 20,
        19: 30,
        39: 40,
        59: 45,
        79: 17,
        99: 56,
        119: 170
    }
    max_y = np.amax(all_array)
    for i, t in enumerate(all_array):
        trace = dict(
            x=times[:1],
            y=t[:1],
            name=i + 1,
            marker=dict(
                color=get_color(i)
            ),
            mode='lines'
        )
        data.append(trace)

        # if i == 0 or ((i + 1) % 20 == 0):
        #     annotation = get_annotation(
        #         trace_index=i,
        #         x=anno_offsets[i],
        #         y_offset=1.2,
        #         trace_data=t,
        #         step_size=time_inc
        #     )
        #     annotations.append(annotation)

    frames = []
    for i in range(1, len(times), int(1 / time_inc)):
        frame_data = []
        for j, t in enumerate(all_array):
            trace = dict(
                x=times[:i],
                y=t[:i],
                name=j + 1,
                marker=dict(
                    color=get_color(j)
                ),
                mode='lines'
            )
            frame_data.append(trace)

        frame = dict(
            data=frame_data
        )
        frames.append(frame)

    layout = dict(
        title='Motor Unit Forces by Time',
        yaxis=dict(
            title='Motor unit force (relative to MU1 tetanus)',
            range=[0, max_y],
            autorange=False
        ),
        xaxis=dict(
            title='Time (s)',
            range=[0, sim_duration],
            autorange=False
        ),
        updatemenus=[{
            'type': 'buttons',
            'buttons': [{
                'args': [
                    None,
                    {'frame': {'duration': 0, 'redraw': False},
                     'mode': 'next',
                     'transition': {'duration': 0, 'easing': 'linear'}
                     }
                ],
                'label': 'Play',
                'method': 'animate'
            }]
        }]
    )
    layout['annotations'] = annotations

    fig = dict(
        data=data,
        layout=layout,
        frames=frames
    )
    plot(fig, filename='animated-forces-by-time.html', validate=False)

if True:
    # 2.B - Per Motor Unit Firing Rate by Time
    all_array = np.array(all_firing_rates).T
    data = []
    annotations = []
    anno_offsets = {
        0: 55,
        19: 60,
        39: 65,
        59: 70,
        79: 80,
        99: 90,
        119: 110
    }
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

        if i == 0 or ((i + 1) % 20 == 0):
            annotation = get_annotation(
                trace_index=i,
                x=anno_offsets[i],
                y_offset=-0.4,
                trace_data=t,
                step_size=time_inc
            )
            annotations.append(annotation)

    layout = go.Layout(
        title='Motor Unit Firing Rates by Time',
        yaxis=dict(
            title='Firing rate (Hz)'
        ),
        xaxis=dict(
            title='Time (s)'
        )
    )
    layout['annotations'] = annotations

    fig = go.Figure(
        data=data,
        layout=layout
    )
    plot(fig, filename='firing-rates-by-time.html')

if True:
    # 2.D - Per Motor Unit Force Capacities by Time
    all_array = np.array(all_capacities).T
    data = []
    annotations = []
    anno_offsets = {
        0: 55,
        19: 60,
        39: 65,
        59: 70,
        79: 80,
        99: 90,
        119: 110
    }
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

        if i == 0 or ((i + 1) % 20 == 0):
            annotation = get_annotation(
                trace_index=i,
                x=anno_offsets[i],
                y_offset=-0.03,
                trace_data=t,
                step_size=time_inc
            )
            annotations.append(annotation)

    layout = go.Layout(
        title='Motor Unit Force Capacities by Time',
        yaxis=dict(
            title='Firing rate (Hz)'
        ),
        xaxis=dict(
            title='Time (s)'
        )
    )
    layout['annotations'] = annotations

    fig = go.Figure(
        data=data,
        layout=layout
    )
    plot(fig, filename='force-capacities-by-time.html')
