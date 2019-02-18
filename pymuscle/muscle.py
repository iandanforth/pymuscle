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
    def max_excitation(self):
        return self._pool.max_excitation

    @property
    def current_forces(self):
        return self._fibers.current_forces

    def step(
        self,
        motor_pool_input: Union[int, float, np.ndarray],
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

        # Ensure we're really passing an ndarray to _pool.step()
        input_as_array = np.array(motor_pool_input)

        motor_pool_output = self._pool.step(input_as_array, step_size)
        return self._fibers.step(motor_pool_output, step_size)


class PotvinFuglevandMuscle(Muscle):
    """
    A thin wrapper around :class:`Muscle <Muscle>` which pre-selects the
    Potvin fiber and motor neuron models.
    """

    def __init__(
        self,
        motor_unit_count: int,
        apply_central_fatigue: bool = True,
        apply_peripheral_fatigue: bool = True
    ):
        pool = PotvinFuglevand2017MotorNeuronPool(
            motor_unit_count,
            apply_fatigue=apply_central_fatigue
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
    A wrapper around :class:`Muscle <Muscle>` which pre-selects the
    Potvin motor neuron model and the PyMuscle specific fiber model.

    In addition this class implements an API (through the primary step()
    method) where inputs to and outputs from the muscle are in the range
    0.0 to 1.0.

    The API for this class is oriented toward use along side physics
    simulations.

    This muscle does *not* include central (motor neuron) fatigue as the
    equations for recovery are not yet available.

    This muscle does include both *peripheral* fatigue and recovery.

    :param max_force: Maximum voluntary isometric force this muscle can produce
        when fully rested and at maximum excitation. (Newtons)
        This along with the force_conversion_factor will determine the number
        of simulated motor units.
    :param force_conversion_factor: This library uses biological defaults to
        convert the desired max_force into a number of motor units which
        make up a muscle. This can result in a large number of motor units
        which may be slow. To improve performance (but diverge from biology)
        you can change this force conversion factor.

        Note: It is likely the default value here will change with major
        versions as better biological data is found.
    """
    def __init__(
        self,
        max_force: float = 32.0,
        force_conversion_factor: float = 0.0123,
        apply_central_fatigue: bool = False,
        apply_peripheral_fatigue: bool = True
    ):

        # Maximum voluntary isometric force this muscle will be able to produce
        self.max_force = max_force

        # Ratio of newtons (N) to internal arbitrary force units
        self.force_conversion_factor = force_conversion_factor

        motor_unit_count = self.force_to_motor_unit_count(
            self.max_force,
            self.force_conversion_factor,
        )

        pool = PotvinFuglevand2017MotorNeuronPool(
            motor_unit_count,
            apply_fatigue=apply_central_fatigue
        )
        fibers = PyMuscleFibers(
            motor_unit_count,
            force_conversion_factor=force_conversion_factor,
            apply_fatigue=apply_peripheral_fatigue
        )

        super().__init__(
            motor_neuron_pool_model=pool,
            muscle_fibers_model=fibers
        )

        # Max output in arbitrary units
        self.max_arb_output = sum(self._fibers._peak_twitch_forces)

    @staticmethod
    def force_to_motor_unit_count(
        max_force: float,
        conversion_factor: float
    ) -> int:
        # This takes the relationship between force production
        # and number of motor units from Fuglevand 93 and solves
        # for motor units given desired force.

        # The reference muscle is the first dorsal interossei
        # https://en.wikipedia.org/wiki/Dorsal_interossei_of_the_hand

        # The number of motor units for the FDI was estimated by
        # Feinstein et al. (1955) at 119.
        # https://www.ncbi.nlm.nih.gov/pubmed/14349537
        #
        # Caveats:
        #  - This estimate was from 1 dissection of an adult male
        #  - This esimate counted 'large nerve fibers' and then assumed
        #    40% of them would be afferent.

        # The Fuglevand experiments use 120 motor units.  The peak twitch
        # forces are calculated in the range of 0-100 arbitrary force units
        # with each motor unit being assigned a peak twitch force according
        # to the exponential function outlined in Fuglevand 93.
        #
        # The total possible force units this muscle could produce, after
        # assignment of peak twitch forces, is the sum of those peak twitch
        # forces or ~2609.03 arbitrary units.

        # The maximum voluntary force (MVC) for the FDI was taken from
        # Jahanmir-Zezhad et al.
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5450824/
        # MVC = 32 Newtons (N)

        # Thus if a modeled FDI with 120 motor units produces an MVC of
        # 2,216 units and a real FDI produces an MVC of 32N we can
        # relate them as follows
        #
        # conversion_factor = 32N / 2609.03 units = 0.01226509469 N/unit

        # See: https://www.desmos.com/calculator/b9xzsaqs1g
        # m = max_force
        # c = conversion_factor
        # u = motor unit count
        # u = 1 / ln( ((1-m/c) / (e^4.6 - m/c)))^(1/4.6) )
        r = (max_force / conversion_factor)
        n2 = 1 - r
        d2 = np.exp(4.6) - r
        inner = np.power((n2 / d2), (1 / 4.6))
        d = np.log(inner)  # This is natural log by default (ln)
        muf = 1 / d
        muc = int(np.ceil(muf))

        return muc

    def get_central_fatigue(self):
        raise NotImplementedError

    def get_peripheral_fatigue(self) -> float:
        """
        Returns fatigue level in the range 0.0 to 1.0 where:

        0.0 - Completely rested
        1.0 - Completely fatigued
        """
        fatigue = 1 - sum(self._fibers.current_peak_forces) / self.max_arb_output
        return fatigue

    def step(
        self,
        motor_pool_input: Union[int, float, np.ndarray],
        step_size: float
    ) -> float:
        """
        Advances the muscle model one step.

        :param motor_pool_input:
            Either a single value or an array of values representing the
            excitatory input to the motor neuron pool for this muscle.
            Range is 0.0 - 1.0.
        :param step_size:
            How far to advance the simulation in time for this step.
        """

        # Rescale the input to the underlying range for the motor pool
        motor_pool_input *= self.max_excitation
        arb_output = super().step(motor_pool_input, step_size)
        # Rescale the output such that it is in the range 0.0 - 1.0
        scaled_output = arb_output / self.max_arb_output
        return scaled_output
