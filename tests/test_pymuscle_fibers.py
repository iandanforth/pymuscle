import numpy as np
import pytest
from copy import copy
from pymuscle import PyMuscleFibers as Fibers


def test_init():
    motor_unit_count = 120
    f = Fibers(120)

    # Check calculated number of motor units
    assert f.motor_unit_count == 120
    assert np.equal(f.current_forces, np.zeros(motor_unit_count)).all()


def test_step():
    motor_unit_count = 120
    f = Fibers(motor_unit_count)

    # Missing arguments
    with pytest.raises(TypeError):
        f.step()

    with pytest.raises(TypeError):
        f.step(33.0)

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


def test_fatigue():
    motor_unit_count = 120
    f = Fibers(motor_unit_count)

    # With zero inputs fatigue shouldn't change
    f = Fibers(motor_unit_count)
    ctf_before = copy(f.current_peak_forces)
    for i in range(100):
        f.step(np.zeros(motor_unit_count), 1.0)

    ctf_after = f.current_peak_forces
    assert np.equal(ctf_before, ctf_after).all()

    # With max input all ctfs should decrease
    f = Fibers(motor_unit_count)
    ctf_before = copy(f.current_peak_forces)
    max_input = 67.0
    for i in range(100):
        f.step(np.full(motor_unit_count, max_input), 1.0)

    ctf_after = f.current_peak_forces
    assert np.greater(ctf_before, ctf_after).all()

    # With fatigue off no ctfs should change
    f = Fibers(motor_unit_count, apply_fatigue=False)
    max_input = 67.0
    for i in range(100):
        f.step(np.full(motor_unit_count, max_input), 1.0)

    ctf_after = f.current_peak_forces
    assert np.equal(ctf_before, ctf_after).all()
