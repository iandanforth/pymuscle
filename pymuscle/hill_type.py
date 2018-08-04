"""
Implementations of equations used in Hill-Type muscle models

Not all equations for a traditional Hill-Type model are provided here. It is
expected that passive elements will be modeled by a physics simulator selected
by PyMuscle users.

Also it should be noted that these equations do not take pennetation angle
into account (which is common in Hill-type models). Users should check to
make sure their physics simulator is handling forces correctly to provide values
which vary by angle.
"""
import numpy as np


def contractile_element_force_length_curve(
    rest_length: float,
    current_length: float,
    curve_width_factor: float=17.33,
    peak_force_length: float=1.1
) -> float:
    """
    This normalizes length to the resting length of the tendon
    Anderson and others normalize by the length that would generate the
    most force. This is a gaussian-like relationship.

    :param rest_length: The resting length of the muscle
    :param current_length: The current length of the muscle
    :param curve_width_factor:
        A curve shape paramater which describes where force begins to approach
        0 for muscles shorter or longer than resting length.
        See https://www.desmos.com/calculator/qosweyyqfk for graph.
    :param peak_force_length:
        The ratio of the length of the muscle when it can generate maximum
        contractile force (sometimes known as the optimal length) and the
        resting length of the muscle.
    """
    norm_length = current_length / rest_length
    peak_force_length = 1.10  # Aubert 1951
    exponent = curve_width_factor * (-1 * (abs(norm_length - peak_force_length) ** 3))
    percentage_of_max_force = np.exp(exponent)
    return percentage_of_max_force


def contractile_element_force_velocity_curve(
    rest_length: float,
    current_length: float,
    prev_length: float,
    time_step: float,
    max_velocity: float = 3.0,
    max_exccentric_multiple: float = 1.8
) -> float:
    """
    Returns a value (0 < value < max_eccentric_multiple). This is a sigmoidal
    relationship.

    :param rest_length: The resting length of the muscle
    :param current_length: The current length of the muscle

    See https://www.desmos.com/calculator/gkcdsdcuyh for graph
    """
    velocity = ((prev_length - current_length) / rest_length) / time_step
    norm_velocity = velocity / max_velocity
    exponent = (0.04 - norm_velocity) / 0.18  # Values to make (0, 1)
    denominator = 1 + np.exp(exponent)
    force_multiple = max_exccentric_multiple - (max_exccentric_multiple / denominator)
    return force_multiple
