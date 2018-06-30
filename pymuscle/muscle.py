import numpy as np
from .potvin_fiber_model import PotvinMuscleFiberModel
from .potvin_motor_neuron_pool import PotvinMotorNeuronPool


class Space(object):

    def __init__(self, shape):
        if isinstance(shape, int):
            shape = [shape]
        self.shape = shape

    def sample(self):
        return np.random.rand(*self.shape)


class Muscle(object):
    """
    A user-created :class:`Muscle <Muscle>` object.

    Used to simulate the input-output relationship between motor neuron
    excitation and muscle fiber contractile state over time.

    :param motor_unit_count: How many motor units comprise this muscle.
    :param history_size: The window over which we calculate fatigues

    Usage::

      from pymuscle import Muscle
      muscle = Muscle(60)
      input = np.rand(60,1)
      muscle.step(input, 1 / 50)
    """

    def __init__(self, motor_unit_count):
        self.motor_unit_count = motor_unit_count
        self.input_space = Space([motor_unit_count])
        self.fiber_output = np.zeros(motor_unit_count)

        self._fibers = PotvinMuscleFiberModel(motor_unit_count)
        self._pool = PotvinMotorNeuronPool(motor_unit_count)

    def step(self, motor_neuron_input: np.ndarray, step_size: float) -> None:
        self.fiber_output = motor_neuron_input * 2
        self.total_fiber_output = sum(self.fiber_output)
