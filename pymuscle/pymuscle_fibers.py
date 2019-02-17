import numpy as np
from numpy import ndarray

from .potvin_fuglevand_2017_muscle_fibers import PotvinFuglevand2017MuscleFibers


class PyMuscleFibers(PotvinFuglevand2017MuscleFibers):
    """
    Encapsulates the muscle fibers portions of the motor unit model. Currently
    a thin wrapper to implement fiber recovery.

    This is the standard muscle fiber model for PyMuscle. Which underlying
    model it extends is expected to change over time. This class exists to
    support the needs of users in the machine learning community. Users who
    belong to the physiology, biomechanical, or medical communities should
    use one of the base classes (such as PotvinFuglevand2017MuscleFibers) as
    those are strict implementations of published work.

    :param motor_unit_count: Number of motor units in the muscle
    :param force_conversion_factor: The ratio of newtons to arbitrary force
        units. All peak twitch forces are calculated internally to lie in a
        range of 0 to 100 arbitrary force units. The maximum force these
        fibers can theoretically produce is the sum of those peak twitch forces.
        To relate the arbitrary force units to SI units you need to provide
        a conversion factor. Increasing this value is essentially saying that
        a given motor unit produces more force than the default value would
        suggest.
    :param max_twitch_amplitude: Max twitch force within the pool
    :param max_contraction_time:
        [milliseconds] Maximum contraction time for a motor unit
    :param contraction_time_range:
        The scale between the fastest contraction time and the slowest
    :fatigue_factor_first_unit:
        The nominal fatigability of the first motor unit in percent / second
    :fatigability_range:
        The scale between the fatigability of the first motor unit and the last
    :contraction_time_change_ratio:
        For each percent of force lost during fatigue, what percentage should
        contraction increase?

    .. todo::
        The argument naming isn't consistent. Sometimes we use 'max' and other
        times we use 'last unit'. Can these be made consistent?

    Usage::

      from pymuscle import PyMuscleFibers as Fibers

      motor_unit_count = 60
      fibers = Fibers(motor_unit_count)
      motor_neuron_firing_rates = np.rand(motor_unit_count) * 10.0
      step_size = 0.01
      force = fibers.step(motor_neuron_firing_rates, step_size)
    """
    def __init__(
        self,
        *args,
        force_conversion_factor: float=0.028,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Ratio of newtons (N) to internal arbitrary force units
        self.force_conversion_factor = force_conversion_factor

        # Define recovery rates
        # Averaged from data in Liu et al. 2002, Table 2
        max_recovery_rate = self._max_fatigue_rate / 2.53
        # Recovery should ~= fatigue for small units, <= for medium units and
        # << for largest units
        # Re-uses the same method as calculating fatigabilities.
        recovery_range = max_recovery_rate / self._nominal_fatigabilities[0]
        self._recovery_rates = self._calc_nominal_fatigabilities(
            self.motor_unit_count,
            recovery_range,
            max_recovery_rate,
            self._peak_twitch_forces
        )

    def _update_fatigue(
        self,
        normalized_forces: ndarray,
        step_size: float
    ) -> None:
        """
        Updates current twitch forces and contraction times. This overrides
        the parent method to add in recovery calculations.

        :param normalized_forces:
            Array of scaled forces. Used to weight how much fatigue will be
            generated in this step.
        :param step_size: How far time has advanced in this step.
        """
        fatigues = (self._nominal_fatigabilities * normalized_forces) * step_size
        self._current_peak_forces -= fatigues

        # Apply recovery for units producing no force
        self._apply_recovery(normalized_forces, step_size)

        # Zero out negative values
        self._current_peak_forces[self._current_peak_forces < 0] = 0.0

        # Clip max values
        over = self._current_peak_forces > self._peak_twitch_forces
        self._current_peak_forces[over] = self._peak_twitch_forces[over]

        # Apply fatigue to contraction times
        self._update_contraction_times()

    def _apply_recovery(
        self,
        normalized_forces: ndarray,
        step_size: float
    ) -> None:
        """
        Apply recovery to motor units not producing force in this step.

        TODO - Finalize the strategy used below
        """
        # Find the indices of valid, recovering units.
        recovering = normalized_forces <= 0

        # Strategy 1 - Linear recovery at fatigue rates
        # recovery = self._nominal_fatigabilities[recovering] * step_size

        # Strategy 2 - Uses inverse of fatigue rates in an asymptotic approach
        # peak = self._peak_twitch_forces[recovering]
        # current = self._current_peak_forces[recovering]
        # recovery_ratio = (peak - current) / peak
        # recovery = (self._nominal_fatigabilities[recovering] * recovery_ratio) * step_size

        # Strategy 3 - Uses linear approach with calculated recovery rates
        # recovery = self._recovery_rates[recovering] * step_size

        # Strategy 4 - Combine 2 and 3
        peak = self._peak_twitch_forces[recovering]
        current = self._current_peak_forces[recovering]
        recovery_ratio = (peak - current) / peak
        recovery = (self._recovery_rates[recovering] * recovery_ratio) * step_size

        self._current_peak_forces[recovering] += recovery
