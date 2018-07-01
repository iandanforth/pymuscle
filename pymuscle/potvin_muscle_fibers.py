import numpy as np
import plotly.graph_objs as go

from plotly.offline import plot
from numpy import ndarray
from copy import copy

from .model import Model

DEBUG = False


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
        max_recruitment_threshold: int = 50
    ):
        super().__init__()

        # Assign public attributes
        self.motor_unit_count = motor_unit_count

        # Calculate the peak twitch force for each motor unit
        motor_unit_indices = np.arange(1, self.motor_unit_count + 1)
        t_log = np.log(max_twitch_amplitude)
        t_exponent = (t_log * (motor_unit_indices)) / (self.motor_unit_count)
        self._peak_twitch_forces = np.exp(t_exponent)

        if DEBUG:
            fig = go.Figure(
                data=[go.Scatter(
                    x=motor_unit_indices,
                    y=self._peak_twitch_forces
                )],
                layout=go.Layout(
                    title='Peak Twitch Forces'
                )
            )
            plot(fig, filename='ptf')

        # Calculate the contraction times for each motor unit
        # Results in a smooth range from max_contraction_time at the first
        # motor unit down to max_contraction_time / contraction_time range
        # for the last motor unit
        scale = np.log(max_twitch_amplitude) / np.log(contraction_time_range)
        self._contraction_times = max_contraction_time * np.power(
            1 / self._peak_twitch_forces,
            1 / scale
        )

        if DEBUG:
            fig = go.Figure(
                data=[go.Scatter(
                    x=motor_unit_indices,
                    y=self._contraction_times
                )],
                layout=go.Layout(
                    title='Contraction Times (ms)'
                )
            )
            plot(fig, filename='ct')

    def _normalize_firing_rates(self, firing_rates: ndarray) -> ndarray:
        # Divide by 1000 here as firing rates are per second where contraction
        # times are in milliseconds.
        return (firing_rates / 1000) * self._contraction_times

    @staticmethod
    def _calc_normalized_forces(normalized_firing_rates: ndarray) -> ndarray:
        normalized_forces = copy(normalized_firing_rates)
        linear_threshold = 0.4  # Values are non-linear above this value
        below_thresh_indices = normalized_forces <= linear_threshold
        above_thresh_indices = normalized_forces > linear_threshold
        normalized_forces[below_thresh_indices] *= 0.3
        exponent = -2 * np.power(
            normalized_forces[above_thresh_indices],
            3
        )
        normalized_forces[above_thresh_indices] = 1 - np.exp(exponent)

        return normalized_forces

    def _calc_inst_forces(self, normalized_force: ndarray) -> ndarray:
        """
        Scales the normalized forces for each motor unit by their peak 
        twitch forces
        """
        return normalized_force * self._peak_twitch_forces

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
        normalized_firing_rates = self._normalize_firing_rates(firing_rates)
        normalized_forces = self._calc_normalized_forces(normalized_firing_rates)
        inst_forces = self._calc_inst_forces(normalized_forces)
        return self._calc_total_inst_force(inst_forces)

    def step(self, motor_pool_output: ndarray) -> float:
        """
        Advance the muscle fibers simulation one step.

        Returns the total instantaneous force produced by all fibers for
        the given input from the motor neuron pool.
        """
        return self._calc_total_fiber_force(motor_pool_output)
