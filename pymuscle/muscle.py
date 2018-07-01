import numpy as np
from typing import Union
try:
    from .potvin_muscle_fibers import PotvinMuscleFibers
    from .potvin_motor_neuron_pool import PotvinMotorNeuronPool
    from .model import Model
except ModuleNotFoundError:
    from model import Model
    from potvin_muscle_fibers import PotvinMuscleFibers
    from potvin_motor_neuron_pool import PotvinMotorNeuronPool


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
    excitation and muscle fibers contractile state over time.

    :param motor_neuron_pool_model:
        The motor neuron pool implementation to use with this muscle.
    :param muscle_fibers_model:
        The muscle fibers model implementation to use with this muscle.
    :param motor_unit_count: How many motor units comprise this muscle.

    Usage::

        from pymuscle import Muscle, PotvinMuscleFibers, PotvinMotorNeuronPool

        motor_unit_count = 60
        muscle = Muscle(
            PotvinMotorNeuronPool(motor_unit_count),
            PotvinMuscleFibers(motor_unit_count),
        )
        excitation = 32.0
        force = muscle.step(excitation, 1 / 50.0)
    """

    def __init__(
        self,
        motor_neuron_pool_model: Model,
        muscle_fibers_model: Model,
    ):
        self._pool = motor_neuron_pool_model
        self._fibers = muscle_fibers_model

    def step(
        self,
        motor_pool_input: Union[float, np.ndarray],
        step_size: float
    ) -> float:
        """
        Advances the muscle model one step.

        :param motor_pool_input:
            Either a single value or an array of values representing the
            excitatory input to the motor neuron pool for this muscle
        :param step_size:
            How far to advance the simulation in time for this step.
        """

        # Expand a single input to the muscle to a full array
        if isinstance(motor_pool_input, float):
            motor_pool_input = np.full(
                self._pool.motor_unit_count,
                motor_pool_input
            )

        motor_pool_output = self._pool.step(motor_pool_input)
        return self._fibers.step(motor_pool_output)


class PotvinMuscle(Muscle):
    """
    A thin wrapper around :class:`Muscle <Muscle>` which pre-selects the
    Potvin fiber and motor neuron models.
    """

    def __init__(
        self,
        motor_unit_count: int,
        pre_calc_firing_rates: bool = False
    ):
        pool = PotvinMotorNeuronPool(
            motor_unit_count,
            pre_calc_firing_rates=pre_calc_firing_rates
        )
        fibers = PotvinMuscleFibers(motor_unit_count)

        super().__init__(
            motor_neuron_pool_model=pool,
            muscle_fibers_model=fibers
        )


if __name__ == '__main__':
    muscle = PotvinMuscle(120, True)

    # Performance Benchmarking
    import time
    start = time.time()
    iterations = 10000
    step_size = 1 / 50.0
    for _ in range(iterations):
        excitation = np.random.random_integers(1, 60) / 1.0  # Quick cast
        force = muscle.step(excitation, step_size)
    duration = time.time() - start
    avg = duration / iterations

    multiple = 100
    real = 1.0 / 60.0
    x_real = real / multiple

    print("{} iterations took {} seconds. {} per iteration".format(
        iterations, duration, avg)
    )

    if avg < x_real:
        print("This is better than {}x real time :)".format(multiple))
    else:
        print("This is worse than {}x real time. :(".format(multiple))

    print("Multiple:", real / avg)
