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
if False:

    # Recruitment Thresholds
    if True:
        fig = go.Figure(
            data=[go.Scatter(
                x=motor_unit_indices,
                y=pool._recruitment_thresholds
            )],
            layout=go.Layout(
                title='Recruitment Thresholds'
            )
        )
        plot(fig, filename='recruitment-thresholds.html')

    # Peak Firing Rates
    if False:
        fig = go.Figure(
            data=[go.Scatter(
                x=motor_unit_indices,
                y=pool._peak_firing_rates
            )],
            layout=go.Layout(
                title='Peak Firing Rates'
            )
        )
        plot(fig, filename='peak-firing-rates.html')

    # Firing Rates by Excitation
    if False:
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
        plot(fig, filename='firing-rates-by-excitation.html')

    # Adaptations
    if True:
        # Uses example values from paper for verification
        excitation = 20.0
        current_time = 15
        excitations = np.full(motor_unit_count, excitation)
        firing_rates = pool._calc_firing_rates(excitations)
        adaptations = pool._calc_adaptations(firing_rates, current_time)
        adapted_firing_rates = pool._calc_adapted_firing_rates(
            excitations,
            current_time
        )

        fig = go.Figure(
            data=[go.Scatter(
                x=motor_unit_indices,
                y=adaptations
            )],
            layout=go.Layout(
                title='Adaptations @ {}'.format(excitation)
            )
        )
        plot(fig, filename='adaptations.html')

        fig = go.Figure(
            data=[go.Scatter(
                x=motor_unit_indices,
                y=adapted_firing_rates
            )],
            layout=go.Layout(
                title='Adapted Firing Rates @ {}'.format(excitation)
            )
        )
        plot(fig, filename='adapted-firing-rates.html')

# Muscle Fiber Charts
if False:
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
    plot(fig, filename='peak-twitch-forces.html')

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
    plot(fig, filename='contraction-times.html')

    # Fig 1.B
    rel_twitch_force = fibers._peak_twitch_forces / fibers._peak_twitch_forces[0]
    fig = go.Figure(
        data=[go.Scatter(
            x=fibers._contraction_times,
            y=rel_twitch_force,
            mode='markers'
        )],
        layout=go.Layout(
            title='Relative Contraction Times'
        )
    )
    plot(fig, filename='relative-contraction-times.html')

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
    plot(fig, filename='nominal-fatigabilities.html')

    # Fig 1.D
    rel_twitch_force = fibers._peak_twitch_forces / fibers._peak_twitch_forces[0]
    fig = go.Figure(
        data=[go.Scatter(
            x=rel_twitch_force,
            y=fibers._nominal_fatigabilities,
            mode='markers'
        )],
        layout=go.Layout(
            title='Relative Fatigabilities'
        )
    )
    plot(fig, filename='relative-fatigabilities.html')

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
    plot(fig, filename='total-force-by-excitation.html')

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
    plot(fig, filename='forces-by-excitation.html')

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
    plot(fig, filename='normalized-forces-by-excitation.html')
