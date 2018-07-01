import sys
import os

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
    120,
    100,
    max_contraction_time=100,
    contraction_time_range=5
)
