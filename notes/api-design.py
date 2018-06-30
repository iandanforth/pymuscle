import numpy as np
from numpy import ndarray


class MotorNeuronPoolModel(object):
    """
    Encapsulates the motor neuron portion of the motor unit model.
    """
    def __init__(
        self,
        motor_unit_count: int,
        activation_resolution: int = 2,
        recruitment_thresh_range: int = 50,
        fatigue_rate_range: int = 180,
        fatigue_factor: float = 0.0225,
        tau: int = 22,  # No idea what this is yet.
        adaptSF: float = 0.67,  # No idea what this is yet.
        ctSF: float = 0.379,  # No idea what this is yet.
        minimum_recruitment_thresh: int = 1,
        recruitment_gain: int = 1,
        min_firing_rate: int = 8,
        first_peak_firing_rate: int = 35,
        last_peak_firing_rate: int = 25,
        firing_rate_slope: int = 1,
    ):
        pass

    def _calc_motor_neuron_output(
        step_size: float,
        motor_neuron_intrinsics: ndarray,
        motor_neuron_fatigue: ndarray,
        motor_neuron_input: ndarray
    ) -> ndarray:
        """
        Returns the output of the motor neuron pool for the next step
        """
        pass

    def _calc_motor_neuron_fatigue(
        step_size: float,
        motor_neuron_intrinsics: ndarray,
        motor_neuron_fatigue: ndarray,
        motor_neuron_output_history: ndarray
    ) -> ndarray:
        pass

    def _update_motor_neuron_output_history(
        motor_neuron_output_history: ndarray,
        motor_neuron_output: ndarray
    ) -> ndarray:
        pass

    def step(
        self,
        step_size: float,
        motor_neuron_input: ndarray,
        motor_neuron_intrinsics: ndarray,
        motor_neuron_fatigue: ndarray,
        motor_neuron_output_history: ndarray,
    ) -> tuple:
        motor_neuron_output = self._calc_motor_neuron_output(
            step_size,
            motor_neuron_intrinsics,
            motor_neuron_fatigue,
            motor_neuron_input
        )
        motor_neuron_output_history = self._update_motor_neuron_output_history(
            motor_neuron_output_history,
            motor_neuron_output
        )
        motor_neuron_fatigue = self._calc_motor_neuron_fatigue(
            step_size,
            motor_neuron_intrinsics,
            motor_neuron_fatigue,
            motor_neuron_output_history
        )

        return (
            motor_neuron_output,
            motor_neuron_output_history,
            motor_neuron_fatigue
        )


class MuscleFiberModel(object):
    """
    Encapsulates the muscle fiber portions of the motor unit model.
    """
    def __init__(
        self,
        twitch_tension_range: int = 100,
        contraction_time_range: int = 3,
        longest_contract_time: int = 90,
    ):
        # PðiÞ 1⁄4 e1⁄2lnðRPÞði  1Þ=ðn  1Þ 

        

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


class Muscle(object):
    """
    A user-created :class:`Muscle <Muscle>` object.

    Used to simulate the input-output relationship between motor neuron
    excitation and muscle fiber contractile state over time.

    :param motor_unit_count: How many motor units comprise this muscle.
    :param history_size: The window over which we calculate fatigues

    Usage::

      >>> from pymuscle import Muscle
      >>> muscle = Muscle(60)
      >>> input = np.rand(60,1)
      >>> muscle.step(input, 1 / 50)
    """

    def __init__(
        self,
        motor_unit_count: int = 120,
        history_size: int = 60,
    ):
        self._motor_unit_count = motor_unit_count

        # Instantiate our model components
        self.motor_neuron_model = MotorNeuronPoolModel(motor_unit_count)
        self.fiber_model = MuscleFiberModel(motor_unit_count)

        # Define immutable properties of neurons and fibers
        self.motor_neuron_intrinsics = gen_motor_neuron_intrinsics()
        self.fiber_intrinsics = gen_fiber_intrinsics()

        # Prepare state containers
        self.motor_neuron_input = ndarray(motor_unit_count)
        self.motor_neuron_fatigue = ndarray(motor_unit_count)
        self.fiber_fatigue = ndarray(motor_unit_count)

        self._history_size = history_size
        self.motor_neuron_output_history = ndarray(
            motor_unit_count,
            history_size
        )
        self.fiber_output_history = ndarray(
            motor_unit_count,
            history_size
        )

    def step(self, activation: float, step_size: float) -> float:
        """
        Advances the simulated muscle state. Each motor unit is given
        <activation>.

        :param activation: Excitatory input to all motor neurons in the muscle
        :param step_size: The time period to advance the simulation in seconds
                          Often a fraction of a second, e.g. (1/50.0)
        """
        return self._step_with_full_input(
            np.full(self.motor_unit_count, activation),
            step_size
        )

    @property
    def motor_unit_count(self):
        return self._motor_unit_count

    @property
    def history_size(self):
        return self._history_size

    def _step_with_full_input(
        self,
        motor_neuron_input: ndarray,
        step_size: float
    ) -> float:
        """
        Advances the simulated muscle state. Allows for control of individual
        motor units if desired. Each motor unit is given the respective value
        from the array <motor_neuron_input>.

        :param motor_neuron_input: Excitatory input for each motor neuron in
                                   the muscle
        :param step_size: The time period to advance the simulation in seconds
                          Often a fraction of a second, e.g. (1/50.0)
        """

        # Update motor neuron model
        (self.motor_neuron_output,
         self.motor_neuron_output_history,
         self.motor_neuron_fatigue) = self.motor_neuron_model.step(
            step_size,
            motor_neuron_input,
            self.motor_neuron_intrinsics,
            self.motor_neuron_fatigue,
            self.motor_neuron_output_history,
        )

        # Update fiber model
        (self.fiber_output,
         self.fiber_output_history,
         self.fiber_fatigue) = self.fiber_model.step(
            step_size,
            self.motor_neuron_output,
            self.fiber_intrinsics,
            self.fiber_fatigue,
            self.fiber_output_history,
        )

        return self.fiber_output
