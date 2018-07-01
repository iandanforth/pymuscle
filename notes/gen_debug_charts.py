import sys
import os
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot

sys.path.insert(0, os.path.abspath('..'))
from pymuscle import PotvinMuscleFibers
from pymuscle import PotvinMotorNeuronPool

# NOTE: For this script to do anything you have to set DEBUG = True
# in the source code for each of these classes.

# Motor Neuron Pool
motor_unit_count = 120
pool = PotvinMotorNeuronPool(
    motor_unit_count,
    50
)

# Fibers
fibers = PotvinMuscleFibers(
    motor_unit_count,
    100,
    max_contraction_time=100,
    contraction_time_range=5
)

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
