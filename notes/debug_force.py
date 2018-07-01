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
motor_unit_count = 4
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
for i in [18.6, 18.7, 18.8, 18.9]:
    i = round(i, 1)
    excitations = np.full(pool.motor_unit_count, i)
    firing_rates = pool.step(excitations)
    print("Firing rates", firing_rates)
    force = fibers.step(firing_rates)
    print(force)
    forces.append(force)

# fig = go.Figure(
#     data=[go.Scatter(
#         x=xs,
#         y=forces
#     )],
#     layout=go.Layout(
#         title='Total force by Excitation'
#     )
# )
# plot(fig, filename='total-force-by-excitation')
