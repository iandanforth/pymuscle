import numpy as np
import pytest
from pymuscle import PotvinFuglevandMuscle as Muscle


def test_init():
    with pytest.raises(TypeError):
        m = Muscle()

    motor_unit_count = 120
    m = Muscle(motor_unit_count)

    assert m.motor_unit_count == motor_unit_count


def test_step():
    motor_unit_count = 120
    m = Muscle(motor_unit_count)

    with pytest.raises(TypeError):
        m.step()

    with pytest.raises(AssertionError):
        m.step(np.ones(3), 1)

    # No excitation
    output = m.step(np.zeros(motor_unit_count), 1.0)
    assert output == pytest.approx(0.0)

    # Moderate
    m = Muscle(motor_unit_count)
    moderate_input = 40.0
    moderate_output = 1311.86896
    output = m.step(np.full(motor_unit_count, moderate_input), 1.0)
    assert output == pytest.approx(moderate_output)

    # Moderate - single value
    m = Muscle(motor_unit_count)
    output = m.step(moderate_input, 1.0)
    assert output == pytest.approx(moderate_output)

    # Max
    m = Muscle(motor_unit_count)
    max_input = m.max_excitation
    max_output = 2215.98114
    output = m.step(np.full(motor_unit_count, max_input), 1.0)
    assert output == pytest.approx(max_output)

    # Above
    m = Muscle(motor_unit_count)
    output = m.step(np.full(motor_unit_count, max_input + 40), 1.0)
    assert output == pytest.approx(max_output)
