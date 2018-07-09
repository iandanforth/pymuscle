import sys
import os
import pytest
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
)
from pymuscle import Model


def test_init():
    with pytest.raises(TypeError):
        m = Model()

    motor_unit_count = 120
    m = Model(120)

    assert m.motor_unit_count == motor_unit_count


def test_step():
    m = Model(100)

    with pytest.raises(TypeError):
        m.step()

    with pytest.raises(NotImplementedError):
        m.step(20, 1)
