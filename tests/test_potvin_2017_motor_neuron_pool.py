import sys
import os
import numpy as np
import pytest
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)
from pymuscle import Potvin2017MotorNeuronPool as Pool


def test_init():
    # Missing arguments
    with pytest.raises(TypeError):
        p = Pool()

    motor_unit_count = 120
    p = Pool(motor_unit_count)
    assert p.motor_unit_count == motor_unit_count


def test_step():
    motor_unit_count = 120
    p = Pool(motor_unit_count)

    # Missing arguments
    with pytest.raises(TypeError):
        p.step()

    # Bad type
    with pytest.raises(TypeError):
        p.step(33.0, 1)

    # Wrong shape
    with pytest.raises(AssertionError):
        p.step(np.ones(3), 1)

    # No excitation
    output = p.step(np.zeros(motor_unit_count), 1.0)
    assert output == pytest.approx(0.0)

    # Moderate
    p = Pool(motor_unit_count)
    moderate_input = 40.0
    moderate_output = 3491.4571777
    output = p.step(np.full(motor_unit_count, moderate_input), 1.0)
    output_sum = np.sum(output)
    assert output_sum == pytest.approx(moderate_output)

    # Max
    p = Pool(motor_unit_count)
    max_input = 67.0
    max_output = 3894.9753008
    output = p.step(np.full(motor_unit_count, max_input), 1.0)
    output_sum = np.sum(output)
    assert output_sum == pytest.approx(max_output)

    # Above
    p = Pool(motor_unit_count)
    output = p.step(np.full(motor_unit_count, max_input + 40), 1.0)
    output_sum = np.sum(output)
    assert output_sum == pytest.approx(max_output)


def test_precacl_values():
    # Pre calculated values and on-the-fly values should be the same
    motor_unit_count = 120
    p1 = Pool(motor_unit_count)
    p2 = Pool(
        motor_unit_count,
        pre_calc_firing_rates=True
    )
    moderate_input = 40.0
    input_array = np.full(motor_unit_count, moderate_input)

    output1 = p1.step(input_array, 1.0)
    output1_sum = np.sum(output1)

    output2 = p2.step(input_array, 1.0)
    output2_sum = np.sum(output2)

    assert output1_sum == pytest.approx(output2_sum)


def test_fatigue_values():
    motor_unit_count = 120

    p = Pool(motor_unit_count)
    max_input = 67.0
    first_output = p.step(np.full(motor_unit_count, max_input), 1.0)
    first_output_sum = np.sum(first_output)

    # Advance the simulation 10 seconds
    for _ in range(10):
        adapted_output = p.step(np.full(motor_unit_count, max_input), 1.0)
        adapted_output_sum = np.sum(adapted_output)

    # Should have changed
    assert adapted_output_sum != pytest.approx(first_output_sum)

    # To this value
    expected_adapted_output = 3737.1571266
    assert adapted_output_sum == pytest.approx(expected_adapted_output)


def test_fatigue_disabled():
    motor_unit_count = 120

    p = Pool(motor_unit_count, apply_fatigue=False)
    max_input = 67.0
    first_output = p.step(np.full(motor_unit_count, max_input), 1.0)
    first_output_sum = np.sum(first_output)

    # Advance the simulation 10 seconds
    for _ in range(10):
        next_output = p.step(np.full(motor_unit_count, max_input), 1.0)
        next_output_sum = np.sum(next_output)

    # Should NOT have changed
    assert next_output_sum == pytest.approx(first_output_sum)
