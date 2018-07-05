import numpy as np
import math # noqa
from numpy import ndarray
from copy import copy

from .model import Model


class PotvinMuscleFibers(Model):
    """
    Encapsulates the muscle fibers portions of the motor unit model.

    The name of each parameter as it appears in Potvin, 2017 is in parentheses.
    If a parameter does not appear in the paper but does appear in the Matlab
    code, the variable name from the Matlab code is in parentheses.

    :param motor_unit_count: Number of motor units in the muscle (n)
    :param max_twitch_amplitude: Max twitch force within the pool (RP)
    :param max_contraction_time:
        [milliseconds] Maximum contraction time for a motor unit (tL)
    :param contraction_time_range:
        The scale between the fastest contraction time and the slowest (rt)
    :fatigue_factor_first_unit:
        The nominal fatigability of the first motor unit in percent / second
    :fatigability_range:
        The scale between the fatigability of the first motor unit and the last
    :contraction_time_change_ratio:
        For each percent of force lost during fatigue, what percentage should
        contraction increase? Based on Shields et al (1997)

    .. todo::
        The argument naming isn't consistent. Sometimes we use 'max' and other
        times we use 'last unit'. Can these be made consistent?

    Usage::

      from pymuscle import PotvinMuscleFibers

      motor_unit_count = 60
      fibers = PotvinMuscleFibers(motor_unit_count)
      motor_neuron_firing_rates = np.rand(motor_unit_count) * 10.0
      force = fibers.step(motor_neuron_firing_rates)
    """
    def __init__(
        self,
        motor_unit_count: int = 120,
        max_twitch_amplitude: int = 100,
        max_contraction_time: int = 90,
        contraction_time_range: int = 3,
        max_recruitment_threshold: int = 50,
        fatigue_factor_first_unit: float = 0.0125,
        max_fatigue_rate: float = 0.0225,
        fatigability_range: int = 180,
        contraction_time_change_ratio: float = 0.379
    ):
        self._peak_twitch_forces = self._calc_peak_twitch_forces(
            motor_unit_count,
            max_twitch_amplitude
        )

        # These will change with fatigue.
        self._current_twitch_forces = copy(self._peak_twitch_forces)

        self._contraction_times = self._calc_contraction_times(
            max_twitch_amplitude,
            max_contraction_time,
            contraction_time_range,
            self._peak_twitch_forces
        )

        # These will change with fatigue
        self._current_contraction_times = copy(self._contraction_times)

        # The maximum rates at which motor units will fatigue
        self._nominal_fatigabilities = self._calc_nominal_fatigabilities(
            motor_unit_count,
            fatigability_range,
            max_fatigue_rate,
            self._peak_twitch_forces
        )

        # Assing other non-public attributes
        self._contraction_time_change_ratio = contraction_time_change_ratio

        # Assign public attributes
        self.motor_unit_count = motor_unit_count

    def _apply_fatigue(self, normalized_forces: ndarray, step_size: float) -> None:
        """
        Updates current twitch forces and contraction times.
        """
        # Instantaneous fatigue rate
        fatigues = (self._nominal_fatigabilities * normalized_forces) * step_size
        self._current_twitch_forces -= fatigues
        # Zero out negative values
        self._current_twitch_forces[self._current_twitch_forces < 0] = 0.0
        self._update_contraction_times()

    def _update_contraction_times(self):
        """
        Update our current contraction times as a function of our current
        force capacity relative to our peak force capacity.
        From Eq. (11)
        """
        force_loss_pcts = 1 - (self._current_twitch_forces / self._peak_twitch_forces)
        inc_pcts = 1 + self._contraction_time_change_ratio * force_loss_pcts
        self._current_contraction_times = self._contraction_times * inc_pcts

    def _calc_ct_increase_percents(self, force_losses: ndarray):
        """
        Calculates the percent that contraction times should increase for
        a given loss in force.

        :param force_losses: Percentage decrease in force being produced
            by each motor unit
        """
        return self._contraction_time_change_ratio * force_losses

    @staticmethod
    def _calc_contraction_times(
        max_twitch_amplitude: int,
        max_contraction_time: int,
        contraction_time_range: int,
        peak_twitch_forces: ndarray
    ) -> ndarray:
        """
        Calculate the contraction times for each motor unit
        Results in a smooth range from max_contraction_time at the first
        motor unit down to max_contraction_time / contraction_time range
        for the last motor unit
        """

        # Fuglevand 93 version - very slightly different values
        # twitch_force_range = peak_twitch_forces[-1] / peak_twitch_forces[0]
        # scale = math.log(twitch_force_range, contraction_time_range)

        # Potvin 2017 version
        scale = np.log(max_twitch_amplitude) / np.log(contraction_time_range)

        mantissa = 1 / peak_twitch_forces
        exponent = 1 / scale
        return max_contraction_time * np.power(mantissa, exponent)

    @staticmethod
    def _calc_peak_twitch_forces(
        motor_unit_count: int,
        max_twitch_amplitude: int
    ) -> ndarray:
        """
        Calculate the peak twitch force for each motor unit
        """
        motor_unit_indices = np.arange(1, motor_unit_count + 1)
        t_log = np.log(max_twitch_amplitude)
        t_exponent = (t_log * (motor_unit_indices - 1)) / (motor_unit_count - 1)
        return np.exp(t_exponent)

    @staticmethod
    def _calc_nominal_fatigabilities(
        motor_unit_count: int,
        fatigability_range: int,
        max_fatigue_rate: float,
        peak_twitch_forces: ndarray
    ) -> ndarray:
        """
        Calculate *nominal* fatigue factors for each motor unit

        Taken more from the matlab code than the paper.
        """
        motor_unit_indices = np.arange(1, motor_unit_count + 1)
        f_log = np.log(fatigability_range)
        motor_unit_fatigue_curve = np.exp((f_log / (motor_unit_count - 1)) * (motor_unit_indices - 1))
        fatigue_rates = motor_unit_fatigue_curve * (max_fatigue_rate / fatigability_range) * peak_twitch_forces
        return fatigue_rates

    def _normalize_firing_rates(self, firing_rates: ndarray) -> ndarray:
        """
        TODO: Should this be moved into Pool as a static method?

        :param firing_rates: Should be the result of pool._calc_adapted_firing_rates()

        """

        # HACK: Subtracting out the min firing rate here gives me the curves
        # I expect ... but why? See Issue #25
        # firing_rates -= 8
        # firing_rates[firing_rates < 0] = 0.0

        # Divide by 1000 here as firing rates are per second where contraction
        # times are in milliseconds.
        return self._current_contraction_times * (firing_rates / 1000)

    @staticmethod
    def _calc_normalized_forces(normalized_firing_rates: ndarray) -> ndarray:
        """
        Calculate motor unit force, relative to its peak force. Force grows
        in a linear fashion up to 0.4 normalized firing rate and then in a
        sigmoid curve afterward.
        """
        normalized_forces = copy(normalized_firing_rates)
        linear_threshold = 0.4  # Values are non-linear above this value
        below_thresh_indices = normalized_forces <= linear_threshold
        above_thresh_indices = normalized_forces > linear_threshold
        # The next two lines are strange and magical
        # In the paper they are simplified to *= 0.3
        # This is the equivalent of the Matlab code
        normalized_forces[below_thresh_indices] /= 0.4
        normalized_forces[below_thresh_indices] *= 1 - np.exp(-2 * (0.4 ** 3))
        exponent = -2 * np.power(
            normalized_forces[above_thresh_indices],
            3
        )
        normalized_forces[above_thresh_indices] = 1 - np.exp(exponent)
        return normalized_forces

    def _calc_inst_forces(self, normalized_forces: ndarray) -> ndarray:
        """
        Scales the normalized forces for each motor unit by their peak 
        twitch forces
        """
        return normalized_forces * self._peak_twitch_forces

    def _calc_current_forces(self, normalized_forces: ndarray) -> ndarray:
        """
        Scales the normalized forces for each motor unit by their current
        remaining twitch force capacity. i.e. like _calc_inst_forces but
        includes fatigue.
        """
        return normalized_forces * self._current_twitch_forces

    @staticmethod
    def _calc_total_inst_force(inst_forces: ndarray) -> ndarray:
        """
        Returns the sum of all instantaneous forces for the motor units
        """
        return np.sum(inst_forces)

    def _calc_total_fiber_force(self, firing_rates: ndarray) -> ndarray:
        """
        Calculates the total instantaneous force produced by all fibers for
        the given instantaneous firing rates.
        """
        normalized_firing_rates = self._normalize_firing_rates_w_fatigue(firing_rates)
        normalized_forces = self._calc_normalized_forces(normalized_firing_rates)
        inst_forces = self._calc_inst_forces(normalized_forces)
        return self._calc_total_inst_force(inst_forces)

    def step(
        self,
        motor_pool_output: ndarray,
        step_size: float = 0.1
    ) -> float:
        """
        Advance the muscle fibers simulation one step.

        Returns the total instantaneous force produced by all fibers for
        the given input from the motor neuron pool.
        """
        return self._calc_total_fiber_force(motor_pool_output)
