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
pool = PotvinMotorNeuronPool(
    motor_unit_count,
    50
)

# Fibers
fibers = PotvinMuscleFibers(
    motor_unit_count,
    100
)

# Motor Neuron Pool Charts
if True:
    # Recruitment Thresholds
    fig = go.Figure(
        data=[go.Scatter(
            x=motor_unit_indices,
            y=pool._recruitment_thresholds
        )],
        layout=go.Layout(
            title='Recruitment Thresholds'
        )
    )
    plot(fig, filename='recruitment-thresholds')

    # Peak Firing Rates
    fig = go.Figure(
        data=[go.Scatter(
            x=motor_unit_indices,
            y=pool._peak_firing_rates
        )],
        layout=go.Layout(
            title='Peak Firing Rates'
        )
    )
    plot(fig, filename='peak-firing-rates')

    # Firing Rates by Excitation
    pre_calc_max = 70.0
    pre_calc_resolution = 0.1
    resolution_places = len(str(pre_calc_resolution).split(".")[1])
    excitations = np.zeros(motor_unit_count)
    all_firing_rates_by_excitation = []
    excitation_values = np.arange(
        0.0,
        pre_calc_max + pre_calc_resolution,
        pre_calc_resolution
    )

    # This assumes Python 3.6+ which has order preserving dicts
    for k, v in pool._firing_rates_by_excitation.items():
        all_firing_rates_by_excitation.append(
            pool._firing_rates_by_excitation[k]
        )

    all_array = np.array(all_firing_rates_by_excitation).T
    data = []
    for i, t in enumerate(all_array):
        trace = go.Scatter(
            x=excitation_values,
            y=t,
            name=i + 1
        )
        data.append(trace)
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title='Firing rates by excitation values'
        ))
    plot(fig, filename='firing-rates-by-excitation')

# Muscle Fiber Charts
if True:
    # Peak Twitch Forces
    fig = go.Figure(
        data=[go.Scatter(
            x=motor_unit_indices,
            y=fibers._peak_twitch_forces
        )],
        layout=go.Layout(
            title='Peak Twitch Forces'
        )
    )
    plot(fig, filename='peak-twitch-forces')

    # Contraction Times
    fig = go.Figure(
        data=[go.Scatter(
            x=motor_unit_indices,
            y=fibers._contraction_times
        )],
        layout=go.Layout(
            title='Contraction Times (ms)'
        )
    )
    plot(fig, filename='contraction-times')

    # Nominal Fatigabilities
    fig = go.Figure(
        data=[go.Scatter(
            x=motor_unit_indices,
            y=fibers._nominal_fatigabilities
        )],
        layout=go.Layout(
            title='Nominal Fatigabilities'
        )
    )
    plot(fig, filename='nominal-fatigabilities')

# Force Charts
if True:
    xs = np.arange(0.0, 70.0, 0.1)
    forces = []
    all_forces_by_excitation = []
    normalized_all_forces_by_excitation = []
    for i in xs:
        i = round(i, 1)
        excitations = np.full(pool.motor_unit_count, i)
        firing_rates = pool.step(excitations)
        normalized_firing_rates = fibers._normalize_firing_rates(firing_rates)
        normalized_forces = fibers._calc_normalized_forces(normalized_firing_rates)
        inst_forces = fibers._calc_inst_forces(normalized_forces)
        all_forces_by_excitation.append(inst_forces)
        normalized_all_forces_by_excitation.append(
            inst_forces / fibers._peak_twitch_forces
        )
        force = fibers._calc_total_inst_force(inst_forces)
        forces.append(force)

    # Total Force
    fig = go.Figure(
        data=[go.Scatter(
            x=xs,
            y=forces
        )],
        layout=go.Layout(
            title='Total force by Excitation'
        )
    )
    plot(fig, filename='total-force-by-excitation')

    # Per Motor Unit Force
    all_array = np.array(all_forces_by_excitation).T
    data = []
    for i, t in enumerate(all_array):
        trace = go.Scatter(
            x=xs,
            y=t,
            name=i + 1
        )
        data.append(trace)
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title='Motor Unit Forces by Excitation Values'
        ))
    plot(fig, filename='forces-by-excitation')

    # Normalized Per Motor Unit Force
    all_array = np.array(normalized_all_forces_by_excitation).T
    data = []
    for i, t in enumerate(all_array):
        trace = go.Scatter(
            x=xs,
            y=t,
            name=i + 1
        )
        data.append(trace)
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title='Normalized Motor Unit Forces by Excitation Values'
        ))
    plot(fig, filename='normalized-forces-by-excitation')
