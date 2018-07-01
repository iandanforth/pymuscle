import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
from numpy import ndarray

from model import Model

DEBUG = False


class PotvinMotorNeuronPool(Model):
    """
    Encapsulates the motor neuron portion of the motor unit model.

    The name of each parameter as it appears in Potvin, 2017 is in parentheses.
    If a parameter does not appear in the paper but does appear in the
    accompanying Matlab code, the variable name from the Matlab code is used in
    the parentheses.

    :param motor_unit_count: Number of motor units in the muscle (n)
    :param max_recruitment_threshold:
        Max excitation required by a motor unit within the pool before
        firing (RR)
    :param firing_gain:
        The slope of firing rate by excitation above threshold (g)
    :param min_firing_rate:
        The minimum firing rate for a motor neuron above threshold (minR)
    :param max_firing_rate_first_unit:
        Max firing rate for the first motor unit (maxR(1))
    :param max_firing_rate_last_unit:
        Max firing rate for the last motor unit (maxR(last))
    :param pre_calc_firing_rates:
        Whether to build a dict mapping excitation levels to firing rates for
        each motor neuron. This can speed up simulation at the cost of
        additional memory.
    :param pre_calc_resolution:
        Step size for excitation levels to pre-calculate (res)
    :param pre_calc_max: Highest excitation value to pre-calculate

    .. todo::
        Correct usage with .step()

    Usage::

      from pymuscle import PotvinMotorNeuronPool
      motor_unit_count = 60
      pool = PotvinMotorNeuronPool(motor_unit_count)
      excitation = np.full(motor_unit_count, 10.0)
      firing_rates = pool.calc_firing_rates(excitation)
    """
    def __init__(
        self,
        motor_unit_count: int = 120,
        max_recruitment_threshold: int = 50,
        firing_gain: float = 1.0,
        min_firing_rate: int = 8,
        max_firing_rate_first_unit: int = 35,
        max_firing_rate_last_unit: int = 25,
        pre_calc_firing_rates: bool = True,
        pre_calc_resolution: float = 0.1,
        pre_calc_max: float = 70.0
    ):

        super().__init__()
        motor_unit_indices = np.arange(1, motor_unit_count + 1)

        # Calculate recruitment thresholds for each motor neuron
        r_log = np.log(max_recruitment_threshold)
        r_exponent = (r_log * (motor_unit_indices)) / (motor_unit_count)
        self._recruitment_thresholds = np.exp(r_exponent)

        if DEBUG:
            fig = go.Figure(
                data=[go.Scatter(
                    x=motor_unit_indices,
                    y=self._recruitment_thresholds
                )],
                layout=go.Layout(
                    title='Recruitment Thresholds'
                )
            )
            plot(fig, filename='recruitment-thresholds')

        # Calculate peak firing rates for each motor neuron
        firing_rate_range = max_firing_rate_first_unit - max_firing_rate_last_unit
        first_recruitment_thresh = self._recruitment_thresholds[0]
        recruitment_thresh_range = max_recruitment_threshold - first_recruitment_thresh
        temp_thresholds = self._recruitment_thresholds - first_recruitment_thresh
        temp_thresholds /= recruitment_thresh_range
        self._peak_firing_rates = max_firing_rate_first_unit - (firing_rate_range * temp_thresholds)

        if DEBUG:
            fig = go.Figure(
                data=[go.Scatter(
                    x=motor_unit_indices,
                    y=self._peak_firing_rates
                )],
                layout=go.Layout(
                    title='Peak Firing Rates'
                )
            )
            plot(fig, filename='peak-firing-rates')

        # Assign public attributes
        self.motor_unit_count = motor_unit_count

        # Assign internal attributes
        self._max_recruitment_threshold = max_recruitment_threshold
        self._firing_gain = firing_gain
        self._min_firing_rate = min_firing_rate
        self._firing_rates_by_excitation = None

        # Pre-calculate firing rates for all motor neurons across a range of
        # possible excitation levels.
        if pre_calc_firing_rates:
            self._firing_rates_by_excitation = {}
            excitations = np.zeros(motor_unit_count)
            all_firing_rates_by_excitation = []
            excitation_values = np.arange(
                0.0,
                pre_calc_max + pre_calc_resolution,
                pre_calc_resolution
            )
            for i in excitation_values:
                excitations += pre_calc_resolution
                self._firing_rates_by_excitation[i] = self._inner_calc_firing_rates(
                    excitations,
                    self._recruitment_thresholds,
                    self._firing_gain,
                    self._min_firing_rate,
                    self._peak_firing_rates
                )
                all_firing_rates_by_excitation.append(self._firing_rates_by_excitation[i])

            if DEBUG:
                all_array = np.array(all_firing_rates_by_excitation).T
                data = []
                for i, t in enumerate(all_array):
                    trace = go.Scatter(
                        x=excitation_values,
                        y=t,
                        name=i+1
                    )
                    data.append(trace)
                fig = go.Figure(
                    data=data,
                    layout=go.Layout(
                        title='Firing rates by excitation values'
                    ))
                plot(fig, filename='firing-rates-by-excitation')

    @staticmethod
    def _inner_calc_firing_rates(
        excitations: ndarray,
        thresholds: ndarray,
        gain: float,
        min_firing_rate: int,
        peak_firing_rates: ndarray
    ) -> ndarray:

        firing_rates = excitations - thresholds
        firing_rates += min_firing_rate
        below_thresh_indices = firing_rates < min_firing_rate
        firing_rates[below_thresh_indices] = 0
        firing_rates *= gain

        # Check for max values
        above_peak_indices = firing_rates > peak_firing_rates
        firing_rates[above_peak_indices] = peak_firing_rates[above_peak_indices]

        return firing_rates

    def _calc_firing_rates(self, excitations: ndarray) -> ndarray:
        """
        Calculates firing rates on a per motor neuron basis for the given
        array of excitations.
        """
        assert (len(excitations) == len(self._recruitment_thresholds))

        if self._firing_rates_by_excitation:
            excitation = excitations[0]  # TODO - Support variations
            firing_rates = self._firing_rates_by_excitation[excitation]
        else:
            firing_rates = self._inner_calc_firing_rates(
                excitations,
                self._recruitment_thresholds,
                self._firing_gain,
                self._min_firing_rate,
                self._peak_firing_rates
            )

        return firing_rates

    def step(self, motor_pool_input: ndarray) -> ndarray:
        return self._calc_firing_rates(motor_pool_input)


if __name__ == '__main__':
    motor_unit_count = 120
    pool = PotvinMotorNeuronPool(
        motor_unit_count,
        50
    )
