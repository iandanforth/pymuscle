"""
Hopper

Learn to control your muscles to make forward progress!
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from envs import MuscledHopperEnv


def main():
    print("WARNING: WORK IN PROGRESS")
    env = MuscledHopperEnv(apply_fatigue=True)

    # Set up the simulation parameters
    sim_duration = 60  # seconds
    frames_per_second = 50
    step_size = 1 / frames_per_second
    total_steps = int(sim_duration / step_size)

    for i in range(total_steps):
        env.step([10.0] * 4)
        env.render()


if __name__ == '__main__':
    main()
