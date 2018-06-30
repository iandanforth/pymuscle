import numpy as np
from plotly.offline import plot
import plotly.graph_objs as go
from numpy import ndarray


class PotvinMuscleFiberModel(object):
    """
    Encapsulates the muscle fiber portions of the motor unit model.

    The name of each parameter as it appears in Potvin, 2017 is in parentheses.
    If a parameter does not appear in the paper but does appear in the Matlab
    code, the variable name from the Matlab code is in parentheses.

    :param motor_unit_count: Number of motor units in the muscle (n)
    :param max_twitch_amplitude: Max twitch force within the pool (RP)
    :param max_contraction_time:
        [milliseconds] Maximum contraction time for a motor unit (tL)
    :param contraction_time_range:
        The scale between the fastest contraction time and the slowest (rt)

    .. todo::
        Correct usage with .step()

    Usage::

      from pymuscle import PotvinMuscleFiberModel
      motor_unit_count = 60
      fibers = PotvinMuscleFiberModel(motor_unit_count)
    """
    def __init__(
        self,
        motor_unit_count: int = 120,
        max_twitch_amplitude: int = 100,
        max_contraction_time: int = 90,
        contraction_time_range: int = 3,
        max_recruitment_threshold: int = 50,
    ):
        # Calculate the peak twitch force for each motor unit
        motor_unit_indices = np.arange(1, motor_unit_count + 1)
        t_log = np.log(max_twitch_amplitude)
        t_exponent = (t_log * (motor_unit_indices)) / (motor_unit_count)
        self.peak_twitch_forces = np.exp(t_exponent)

        # Calculate the contraction times for each motor unit
        # Results in a smooth range from max_contraction_time at the first
        # motor unit down to max_contraction_time / contraction_time range
        # for the last motor unit
        scale = np.log(max_twitch_amplitude) / np.log(contraction_time_range)
        self.contraction_times = max_contraction_time * np.power(
            1 / self.peak_twitch_forces,
            1 / scale
        )

    def _calc_fiber_output(
        step_size: float,
        fiber_intrinsics: ndarray,
        fiber_fatigue: ndarray,
        motor_neuron_output: ndarray
    ) -> ndarray:
        pass

    def _calc_fiber_fatigue(
        step_size: float,
        fiber_intrinsics: ndarray,
        fiber_fatigue: ndarray,
        fiber_output_history: ndarray
    ) -> ndarray:
        pass

    def _calc_fiber_recovery(
        step_size: float,
        fiber_intrinsics: ndarray,
        fiber_fatigue: ndarray,
        fiber_output_history: ndarray
    ) -> ndarray:
        pass

    def _update_fiber_output_history(
        fiber_output_history: ndarray,
        fiber_output: ndarray
    ) -> ndarray:
        pass

    def step(
        self,
        step_size: float,
        motor_neuron_output: ndarray,
        fiber_intrinsics: ndarray,
        fiber_fatigue: ndarray,
        fiber_output_history: ndarray,
    ) -> tuple:
        fiber_output = self._calc_fiber_output(
            step_size,
            fiber_intrinsics,
            fiber_fatigue,
            motor_neuron_output
        )
        fiber_output_history = self._update_fiber_output_history(
            fiber_output_history,
            fiber_output
        )
        fiber_fatigue = self._calc_fiber_fatigue(
            step_size,
            fiber_intrinsics,
            fiber_fatigue,
            fiber_output_history
        )
        fiber_fatigue = self._calc_fiber_recovery(
            step_size,
            fiber_intrinsics,
            fiber_fatigue,
            fiber_output_history
        )

        return (fiber_output, fiber_output_history, fiber_fatigue)


if __name__ == '__main__':
    fibers = PotvinMuscleFiberModel(
        120,
        100,
        max_contraction_time=100,
        contraction_time_range=5
    )

