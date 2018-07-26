import numpy as np
import pytest
from pymuscle import PotvinFuglevand2017MuscleFibers as Fibers


def test_init():
    # Missing arguments
    with pytest.raises(TypeError):
        f = Fibers()

    motor_unit_count = 120
    f = Fibers(motor_unit_count)
    assert f.motor_unit_count == motor_unit_count


def test_step():
    motor_unit_count = 120
    f = Fibers(motor_unit_count)

    # Missing arguments
    with pytest.raises(TypeError):
        f.step()

    # Bad type
    with pytest.raises(TypeError):
        f.step(33.0, 1)

    # Wrong shape
    with pytest.raises(AssertionError):
        f.step(np.ones(3), 1)

    # No excitation
    output = f.step(np.zeros(motor_unit_count), 1.0)
    assert output == pytest.approx(0.0)

    # Moderate
    f = Fibers(motor_unit_count)
    moderate_input = 40.0
    moderate_output = 2586.7530897
    output = f.step(np.full(motor_unit_count, moderate_input), 1.0)
    output_sum = np.sum(output)
    assert output_sum == pytest.approx(moderate_output)

    # Max
    f = Fibers(motor_unit_count)
    max_input = 67.0
    max_output = 2609.0308816
    output = f.step(np.full(motor_unit_count, max_input), 1.0)
    output_sum = np.sum(output)
    assert output_sum == pytest.approx(max_output)

    # Above
    f = Fibers(motor_unit_count)
    output = f.step(np.full(motor_unit_count, max_input + 40), 1.0)
    output_sum = np.sum(output)
    assert output_sum == pytest.approx(max_output)
