import numpy as np


class Muscle(object):

    def __init__(self, motor_unit_count):
        self.motor_unit_count = motor_unit_count
        self.state_array = np.zeros((1, self.motor_unit_count))
