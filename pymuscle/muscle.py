import numpy as np


class Space(object):

    def __init__(self, shape):
        if isinstance(shape, int):
            shape = [shape]
        self.shape = shape

    def sample(self):
        return np.random.rand(*self.shape)


class Muscle(object):

    def __init__(self, motor_unit_count):
        self.motor_unit_count = motor_unit_count
        self.input_space = Space([motor_unit_count])
        self.fiber_output = np.zeros(motor_unit_count)

    def step(self, motor_neuron_input: np.ndarray, step_size: float) -> None:
        self.fiber_output = motor_neuron_input * 2
        self.total_fiber_output = sum(self.fiber_output)
