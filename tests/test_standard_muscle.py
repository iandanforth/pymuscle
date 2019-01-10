import numpy as np
import pytest
from pymuscle import StandardMuscle as Muscle


def test_init():
    max_force = 32.0
    m = Muscle(max_force)

    # Check calculated number of motor units
    assert m.motor_unit_count == 120


def test_step():
    max_force = 32.0
    m = Muscle(max_force)

    with pytest.raises(TypeError):
        m.step()

    with pytest.raises(AssertionError):
        m.step(np.ones(3), 1)

    # No excitation
    output = m.step(np.zeros(m.motor_unit_count), 1.0)
    assert output == pytest.approx(0.0)

    # Moderate
    m = Muscle(max_force)
    moderate_input = 0.5
    moderate_output = 1020.0358
    output = m.step(np.full(m.motor_unit_count, moderate_input), 1.0)
    assert output == pytest.approx(moderate_output)

    # Moderate - single value
    m = Muscle(max_force)
    output = m.step(moderate_input, 1.0)
    assert output == pytest.approx(moderate_output)

    # Max
    m = Muscle(max_force)
    max_input = 1.0
    max_output = 2215.98114
    output = m.step(np.full(m.motor_unit_count, max_input), 1.0)
    assert output == pytest.approx(max_output)

    # Above
    m = Muscle(max_force)
    output = m.step(np.full(m.motor_unit_count, max_input + 40), 1.0)
    assert output == pytest.approx(max_output)
