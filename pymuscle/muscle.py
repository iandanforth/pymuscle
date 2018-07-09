import numpy as np
from typing import Union

from .potvin_2017_muscle_fibers import Potvin2017MuscleFibers as Fibers
from .potvin_2017_motor_neuron_pool import Potvin2017MotorNeuronPool as Pool
from .model import Model


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

        from pymuscle import (Muscle,
                              Potvin2017MotorNeuronPool as Pool,
                              Potvin2017MuscleFibers as Fibers)

        motor_unit_count = 60
        muscle = Muscle(
            Pool(motor_unit_count),
            Fibers(motor_unit_count),
        )
        excitation = 32.0
        force = muscle.step(excitation, 1 / 50.0)
    """

    def __init__(
        self,
        motor_neuron_pool_model: Model,
        muscle_fibers_model: Model,
    ):
        assert motor_neuron_pool_model.motor_unit_count == \
            muscle_fibers_model.motor_unit_count

        self._pool = motor_neuron_pool_model
        self._fibers = muscle_fibers_model

    @property
    def motor_unit_count(self):
        return self._pool.motor_unit_count

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

        motor_pool_output = self._pool.step(motor_pool_input, step_size)
        return self._fibers.step(motor_pool_output)


class PotvinMuscle(Muscle):
    """
    A thin wrapper around :class:`Muscle <Muscle>` which pre-selects the
    Potvin fiber and motor neuron models.
    """

    def __init__(
        self,
        motor_unit_count: int,
        apply_fatigue: bool = True,
        pre_calc_firing_rates: bool = False
    ):
        pool = Pool(
            motor_unit_count,
            apply_fatigue=apply_fatigue,
            pre_calc_firing_rates=pre_calc_firing_rates
        )
        fibers = Fibers(
            motor_unit_count,
            apply_fatigue=apply_fatigue
        )

        super().__init__(
            motor_neuron_pool_model=pool,
            muscle_fibers_model=fibers
        )
