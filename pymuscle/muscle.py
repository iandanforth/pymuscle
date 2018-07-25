"""
Contains base Muscle class and its immediate descendants.
"""

import numpy as np
from typing import Union

from .potvin_fuglevand_2017_muscle_fibers import PotvinFuglevand2017MuscleFibers
from .potvin_fuglevand_2017_motor_neuron_pool import PotvinFuglevand2017MotorNeuronPool
from .pymuscle_fibers import PyMuscleFibers
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
                              PotvinFuglevand2017MotorNeuronPool as Pool,
                              PotvinFuglevand2017MuscleFibers as Fibers)

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

    @property
    def current_forces(self):
        return self._fibers.current_forces

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
        if isinstance(motor_pool_input, float) or \
           isinstance(motor_pool_input, int):
            motor_pool_input = np.full(
                self._pool.motor_unit_count,
                motor_pool_input
            )

        motor_pool_output = self._pool.step(motor_pool_input, step_size)
        return self._fibers.step(motor_pool_output)


class PotvinFuglevandMuscle(Muscle):
    """
    A thin wrapper around :class:`Muscle <Muscle>` which pre-selects the
    Potvin fiber and motor neuron models.
    """

    def __init__(
        self,
        motor_unit_count: int,
        apply_central_fatigue: bool = True,
        apply_peripheral_fatigue: bool = True,
        pre_calc_firing_rates: bool = False
    ):
        pool = PotvinFuglevand2017MotorNeuronPool(
            motor_unit_count,
            apply_fatigue=apply_central_fatigue,
            pre_calc_firing_rates=pre_calc_firing_rates
        )
        fibers = PotvinFuglevand2017MuscleFibers(
            motor_unit_count,
            apply_fatigue=apply_peripheral_fatigue
        )

        super().__init__(
            motor_neuron_pool_model=pool,
            muscle_fibers_model=fibers
        )


class StandardMuscle(Muscle):
    """
    A thin wrapper around :class:`Muscle <Muscle>` which pre-selects the
    Potvin motor neuron model and the PyMuscle specific fiber model.

    It is expected that this will use a motor neuron model specific to PyMuscle
    (to be called the PyMuscleMotorNeuronPool) in the future.

    This muscle does *not* include central (motor neuron) fatigue as the
    equations for recovery are not yet available.

    This muscle does include both peripheral fatigue and recovery.
    """
    def __init__(
        self,
        motor_unit_count: int,
        apply_central_fatigue: bool = False,
        apply_peripheral_fatigue: bool = True,
        pre_calc_firing_rates: bool = False
    ):
        pool = PotvinFuglevand2017MotorNeuronPool(
            motor_unit_count,
            apply_fatigue=apply_central_fatigue,
            pre_calc_firing_rates=pre_calc_firing_rates
        )
        fibers = PyMuscleFibers(
            motor_unit_count,
            apply_fatigue=apply_peripheral_fatigue
        )

        super().__init__(
            motor_neuron_pool_model=pool,
            muscle_fibers_model=fibers
        )
