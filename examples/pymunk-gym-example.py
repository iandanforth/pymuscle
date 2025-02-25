"""
Arm Curl

The goal here is to keep the arm tracking a moving target smoothly.
Watch as the effort required slowly increases for all portions of the motion,
until the muscle is no longer able to support the arm. Here a hand tuned
PID controller takes care of managing this effort.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from envs import PymunkArmEnv


def main():
    env = PymunkArmEnv(apply_fatigue=True)

    # Set up the simulation parameters
    sim_duration = 120  # seconds
    frames_per_second = 50
    step_size = 1 / frames_per_second
    total_steps = int(sim_duration / step_size)

    # Here we are going to send a constant excitation to the tricep and
    # vary the excitation of the bicep as we try to hit a target location.
    brachialis_input = 0.4  # Percent of max input
    tricep_input = 0.2
    hand_target_y = 200
    target_delta = -10   # Negative delta because lower y values are higher on screen
    # Hand tuned PID params.
    Kp = 0.0001
    Ki = 0.00004
    Kd = 0.0001
    prev_y = None
    print("Fraction of max excitation ...")
    for i in range(total_steps):
        hand_x, hand_y = env.step(
            [brachialis_input, tricep_input],
            step_size, debug=False
        )

        # PID Control
        if prev_y is None:
            prev_y = hand_y

        # Proportional component - INVERTED for flipped coordinate system
        # In the flipped system, we need to DECREASE y to move UP
        error = hand_y - hand_target_y  # Inverted error calculation
        alpha = Kp * error
        # Add integral component - signs adjusted for flipped coordinates
        i_c = Ki * (error * step_size)
        alpha += i_c  # Sign changed from - to +
        # Add in differential component - signs adjusted for flipped coordinates
        d_c = Kd * ((prev_y - hand_y) / step_size)  # Inverted velocity calculation
        alpha += d_c  # Sign changed from - to +

        prev_y = hand_y
        brachialis_input += alpha

        if brachialis_input > 1.0:
            brachialis_input = 1.0
        if brachialis_input < 0.0:
            brachialis_input = 0.0

        # Vary our set point and display the excitation required
        if i % frames_per_second == 0:
            print(brachialis_input)
            print(prev_y)
            hand_target_y += target_delta

        # Switch directions every 5 seconds
        if i > 0 and i % (frames_per_second * 5) == 0:
            target_delta *= -1

        env.render()


if __name__ == '__main__':
    main()
