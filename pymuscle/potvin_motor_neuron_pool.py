import numpy as np
from numpy import ndarray

from .model import Model


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
    :param derecruitment_delta:
        Absolute minimum firing rate = min_firing_rate - derecruitment_delta
        (d)
    :param adaptation_magnitude:
        Magnitude of adaptation for different levels of excitation.(phi)
    :param adaptation_time_constant:
        Time constant for motor neuron adaptation (tau). Default based on
        Revill & Fuglevand (2011)

    .. todo::
        Make pre_calc_max a function of other values as in the matlab code.
        This will also require changing how we look up values if they
        are larger than this value.

    Usage::

      from pymuscle import PotvinMotorNeuronPool

      motor_unit_count = 60
      pool = PotvinMotorNeuronPool(motor_unit_count)
      excitation = np.full(motor_unit_count, 10.0)
      firing_rates = pool.step(excitation)
    """
    def __init__(
        self,
        motor_unit_count: int = 120,
        max_recruitment_threshold: int = 50,
        firing_gain: float = 1.0,
        min_firing_rate: int = 8,
        max_firing_rate_first_unit: int = 35,
        max_firing_rate_last_unit: int = 25,
        pre_calc_firing_rates: bool = False,
        pre_calc_resolution: float = 0.1,
        pre_calc_max: float = 70.0,
        derecruitment_delta: int = 2,
        adaptation_magnitude: float = 0.67,
        adaptation_time_constant: float = 22.0,
    ):
        self._recruitment_thresholds = self._calc_recruitment_thresholds(
            motor_unit_count,
            max_recruitment_threshold
        )

        self._peak_firing_rates = self._calc_peak_firing_rates(
            max_firing_rate_first_unit,
            max_firing_rate_last_unit,
            max_recruitment_threshold,
            self._recruitment_thresholds
        )

        # TODO should have non-numeric value if not recruited
        self._recruitment_times = np.zeros(motor_unit_count)
        self._recruitment_durations = np.zeros(motor_unit_count)

        # Assign additional non-public attributes
        self._max_recruitment_threshold = max_recruitment_threshold
        self._firing_gain = firing_gain
        self._min_firing_rate = min_firing_rate
        self._derecruitment_delta = derecruitment_delta
        self._adaptation_magnitude = adaptation_magnitude
        self._adaptation_time_constant = adaptation_time_constant

        # Assign public attributes
        self.motor_unit_count = motor_unit_count

        # Pre-calculate firing rates for all motor neurons across a range of
        # possible excitation levels.
        self._firing_rates_by_excitation = None
        if pre_calc_firing_rates:
            self._build_firing_rates_cache(pre_calc_max, pre_calc_resolution)

    def _build_firing_rates_cache(
        self,
        pre_calc_max: float,
        pre_calc_resolution: float
    ) -> None:
        """
        Pre-calculate and store firing rates for all motor neurons across a
        range of possible excitation levels.
        """

        self._firing_rates_by_excitation = {}
        # TODO: This is a hack. Maybe memoize vs pre-calculate?
        # Maybe https://docs.python.org/3/library/functools.html#functools.lru_cache
        resolution_places = len(str(pre_calc_resolution).split(".")[1])
        excitations = np.zeros(self.motor_unit_count)
        excitation_values = np.arange(
            0.0,
            pre_calc_max + pre_calc_resolution,
            pre_calc_resolution
        )
        for i in excitation_values:
            i = round(i, resolution_places)
            excitations += pre_calc_resolution
            self._firing_rates_by_excitation[i] = \
                self._inner_calc_firing_rates(
                    excitations,
                    self._recruitment_thresholds,
                    self._firing_gain,
                    self._min_firing_rate,
                    self._peak_firing_rates
            )

    def _calc_adapted_firing_rates(
        self,
        excitations: ndarray,
        current_time: float
    ) -> ndarray:
        """
        Calculate the firing rate for the given excitation including motor
        neuron fatigue (adaptation).
        """
        firing_rates = self._calc_firing_rates(excitations)
        self._update_recruitment_durations(firing_rates, current_time)
        adaptations = self._calc_adaptations(firing_rates)

        adapted_firing_rates = firing_rates - adaptations
        return adapted_firing_rates

    def _calc_firing_rates(self, excitations: ndarray) -> ndarray:
        """
        Calculates firing rates on a per motor neuron basis for the given
        array of excitations.
        """
        assert (len(excitations) == len(self._recruitment_thresholds))

        # if self._firing_rates_by_excitation:
        #     excitation = excitations[0]  # TODO - Support variations
        #     firing_rates = self._firing_rates_by_excitation[excitation]
        # else:
        firing_rates = self._inner_calc_firing_rates(
            excitations,
            self._recruitment_thresholds,
            self._firing_gain,
            self._min_firing_rate,
            self._peak_firing_rates
        )

        return firing_rates

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

    def _update_recruitment_durations(
        self,
        firing_rates: ndarray,
        current_time: float
    ) -> None:
        """
        Increment the on duration for each on motor unit by step_size
        TODO: Decay the on duration or reset it after some period
        """
        indices = (firing_rates > 0) & (self._recruitment_times == 0)
        self._recruitment_times[indices] = current_time
        self._recruitment_durations = current_time - self._recruitment_times
        print(self._recruitment_durations)

    def _calc_adaptations(self, firing_rates: ndarray) -> ndarray:
        adapt_curve = self._calc_adaptations_curve(firing_rates)
        # From Eq. (12)
        exponent = -1 * (self._recruitment_durations / self._adaptation_time_constant)
        adapt_scale = 1 - np.exp(exponent)
        adaptations = adapt_curve * adapt_scale
        # Zero out negative values
        adaptations[adaptations < 0] = 0.0
        return adaptations

    def _calc_adaptations_curve(self, firing_rates: ndarray) -> ndarray:
        """
        Calculates q(i) from Eq. (13)
        """
        ratios = (self._recruitment_thresholds - 1) / (self._max_recruitment_threshold - 1)
        adaptations = self._adaptation_magnitude * (firing_rates - self._min_firing_rate + self._derecruitment_delta) * ratios
        return adaptations

    @staticmethod
    def _calc_peak_firing_rates(
        max_firing_rate_first_unit: int,
        max_firing_rate_last_unit: int,
        max_recruitment_threshold: int,
        recruitment_thresholds: ndarray,
    ) -> ndarray:
        """
        Calculate peak firing rates for each motor neuron

        frdiff = pfr1 - pfrL
        frp = pfr1 - (frdiff * ((recruit_thresh[n] - recruit_thresh[1]) / (r - recruit_thresh[1])))
        """
        firing_rate_range = max_firing_rate_first_unit - max_firing_rate_last_unit
        rates = max_firing_rate_first_unit \
            - (firing_rate_range
                * ((recruitment_thresholds - recruitment_thresholds[0])
                    / (max_recruitment_threshold - recruitment_thresholds[0])))
        return rates

    @staticmethod
    def _calc_recruitment_thresholds(
        motor_unit_count: int,
        max_recruitment_threshold: int
    ) -> ndarray:
        """
        Calculate recruitment thresholds for each motor neuron
        """
        motor_unit_indices = np.arange(1, motor_unit_count + 1)

        r_log = np.log(max_recruitment_threshold)
        r_exponent = (r_log * (motor_unit_indices - 1)) / (motor_unit_count - 1)
        return np.exp(r_exponent)



    def step(self, motor_pool_input: ndarray) -> ndarray:
        """
        Advance the motor neuron pool simulation one step.

        Returns firing rates on a per motor neuron basis for the given
        array of excitations.
        """
        return self._calc_firing_rates(motor_pool_input)
