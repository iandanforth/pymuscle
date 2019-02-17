import numpy as np
import pytest
from pymuscle import StandardMuscle as Muscle


def test_init():
    max_force = 32.0
    m = Muscle(max_force)

    # Check calculated number of motor units
    assert m.motor_unit_count == 120

    # Check calculated max_output
    max_output = 2609.0309
    assert m.max_arb_output == pytest.approx(max_output)

    max_force = 90.0
    m = Muscle(max_force)

    # Check calculated number of motor units
    assert m.motor_unit_count == 340
    max_output = 7338.29062
    assert m.max_arb_output == pytest.approx(max_output)


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
    moderate_output = 0.39096348
    output = m.step(np.full(m.motor_unit_count, moderate_input), 1.0)
    assert output == pytest.approx(moderate_output)

    # Moderate - single value
    m = Muscle(max_force)
    output = m.step(moderate_input, 1.0)
    assert output == pytest.approx(moderate_output)

    # Max
    m = Muscle(max_force)
    max_input = 1.0
    max_output = 0.84935028
    output = m.step(np.full(m.motor_unit_count, max_input), 1.0)
    assert output == pytest.approx(max_output)

    # Above
    m = Muscle(max_force)
    output = m.step(np.full(m.motor_unit_count, max_input + 40), 1.0)
    assert output == pytest.approx(max_output)


def test_fatigue():
    # With fatigue off the return should always be the same
    max_force = 32.0
    m = Muscle(max_force, apply_peripheral_fatigue=False)

    # As measured by fatigue
    fatigue_before = m.get_peripheral_fatigue()
    assert fatigue_before == 0.0

    # As measured by output
    moderate_input = 0.5
    moderate_output = 0.39096348
    for i in range(100):
        output = m.step(moderate_input, 1.0)
    assert output == pytest.approx(moderate_output)

    # And again by fatigue after
    fatigue_after = m.get_peripheral_fatigue()
    assert fatigue_after == 0.0

    # Fatigue ON

    # You should see no change with zero activation
    m = Muscle(max_force, apply_peripheral_fatigue=True)

    # As measured by fatigue
    fatigue_before = m.get_peripheral_fatigue()
    assert fatigue_before == 0.0

    for i in range(100):
        output = m.step(0.0, 1.0)
    assert output == pytest.approx(0.0)

    # As measured by fatgue after
    fatigue_after = m.get_peripheral_fatigue()
    assert fatigue_after == 0.0

    # You should see increasing fatigue and decreasing output
    # with non-zero inputs.

    # Measured by output
    fatigued_output = 0.29151827
    for i in range(100):
        output = m.step(moderate_input, 1.0)
    assert output == pytest.approx(fatigued_output)

    # As measured by fatgue after
    expected_fatigue = 0.18028181
    fatigue_after = m.get_peripheral_fatigue()
    assert pytest.approx(fatigue_after, expected_fatigue)
