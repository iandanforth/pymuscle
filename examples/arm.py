import sys
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
sys.path.append("..")
from pymuscle import Muscle


def main():
    env = ArmEnvironment()
    step_size = 1 / 50.0

    sim_duration = 60  # seconds
    total_steps = int(sim_duration / step_size)
    steps = 0
    input_delta = 5
    brachialis_input = np.ones(5) * 20
    tricep_input = np.ones(5) * 60
    for _ in range(total_steps):
        steps += 1
        if steps % 50 == 0:
            brachialis_input += input_delta

        if steps % 1000 == 0:
            input_delta *= -1

        all_input = np.concatenate((brachialis_input, tricep_input))
        env.step(all_input, step_size, debug=False)
        env.render()


class Space(object):

    def __init__(self, shape):
        if isinstance(shape, int):
            shape = [shape]
        self.shape = shape

    def sample(self):
        return np.random.rand(*self.shape)


class ArmEnvironment(object):
    """
    An environment contains everything external to the agent.

    This class is a simplified version of an OpenAI Gym environment.
    """
    def __init__(self):
        # Set up our 2D physics simulation
        self._init_sim()

        # Add a simulated arm consisting of:
        #  - bones (rigid bodies)
        #  - muscle bodies (damped spring constraints)
        self.brach, self.tricep = self._add_arm()

        # Instantiate the PyMuscles
        brach_motor_unit_count = 5
        self.brach_muscle = Muscle(brach_motor_unit_count)
        tricep_motor_unit_count = 5
        self.tricep_muscle = Muscle(tricep_motor_unit_count)

        # Provide a combined space to define a valid input
        total_space = brach_motor_unit_count + tricep_motor_unit_count
        self.input_space = Space(total_space)

        self.frames = 0
        self.stiffness_delta = 50

    def _init_sim(self):
        pygame.init()
        screen_width = screen_height = 600
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Curl Sim")
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options.flags = 1  # Disable constraint drawing
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -980.0)

    def _add_arm(self):
        config = {
            "arm_center": (self.screen.get_width() / 2,
                           self.screen.get_height() / 2),
            "lower_arm_length": 170,
            "lower_arm_starting_angle": 15,
            "lower_arm_mass": 10,
            "brach_rest_length": 5,
            "brach_stiffness": 450,
            "brach_damping": 200,
            "tricep_rest_length": 30,
            "tricep_stiffness": 50,
            "tricep_damping": 400
        }

        # Upper Arm
        upper_arm_length = 200
        upper_arm_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        upper_arm_body.position = config["arm_center"]
        upper_arm_body.angle = np.deg2rad(-45)
        upper_arm_line = pymunk.Segment(upper_arm_body, (0, 0), (-upper_arm_length, 0), 5)
        upper_arm_line.sensor = True  # Disable collision

        self.space.add(upper_arm_body)
        self.space.add(upper_arm_line)

        # Lower Arm
        lower_arm_body = pymunk.Body(0, 0)  # Pymunk will calculate moment based on mass of attached shape
        lower_arm_body.position = config["arm_center"]
        lower_arm_body.angle = np.deg2rad(config["lower_arm_starting_angle"])
        elbow_extension_length = 20
        lower_arm_start = (-elbow_extension_length, 0)
        lower_arm_line = pymunk.Segment(
            lower_arm_body,
            lower_arm_start,
            (config["lower_arm_length"], 0),
            5
        )
        lower_arm_line.mass = config["lower_arm_mass"]
        lower_arm_line.friction = 1.0

        self.space.add(lower_arm_body)
        self.space.add(lower_arm_line)

        # Pivot (Elbow)
        elbow_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        elbow_body.position = config["arm_center"]
        elbow_joint = pymunk.PivotJoint(elbow_body, lower_arm_body, config["arm_center"])
        self.space.add(elbow_joint)

        # Spring (Brachialis Muscle)
        brach_spring = pymunk.constraint.DampedSpring(
            upper_arm_body,
            lower_arm_body,
            (-(upper_arm_length * (1 / 2)), 0),  # Connect half way up the upper arm
            (config["lower_arm_length"] / 5, 0),  # Connect near the bottom of the lower arm
            config["brach_rest_length"],
            config["brach_stiffness"],
            config["brach_damping"]
        )
        self.space.add(brach_spring)

        # Spring (Tricep Muscle)
        tricep_spring = pymunk.constraint.DampedSpring(
            upper_arm_body,
            lower_arm_body,
            (-(upper_arm_length * (3 / 4)), 0),
            lower_arm_start,
            config["tricep_rest_length"],
            config["tricep_stiffness"],
            config["tricep_damping"]
        )
        self.space.add(tricep_spring)

        # Elbow stop (prevent under/over extension)
        elbow_stop_point = pymunk.Circle(
            upper_arm_body,
            radius=5,
            offset=(-elbow_extension_length, -3)
        )
        elbow_stop_point.friction = 1.0
        self.space.add(elbow_stop_point)

        return brach_spring, tricep_spring

    @staticmethod
    def _handle_keys():
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)

    def step(self, input_array, step_size, debug=True):
        # Check for user input
        self._handle_keys()

        if debug:
            print(input_array)

        # Advance the simulation
        self.space.step(step_size)
        self.frames += 1

        # Advance muscle sim and sync with physics sim
        self.brach_muscle.step(input_array[:5], step_size)  # TODO: Super janky fix this
        brach_output = self.brach_muscle.total_fiber_output
        self.tricep_muscle.step(input_array[5:], step_size)
        tricep_output = self.tricep_muscle.total_fiber_output

        if debug:
            print("Brach Total Output: ", brach_output)
            print("Tricep Total Output: ", tricep_output)

        self.tricep.stiffness = tricep_output
        self.brach.stiffness = brach_output

    def render(self, debug=True):
        if debug and (self.draw_options.flags is not 3):
            self.draw_options.flags = 3  # Enable constraint drawing

        self.screen.fill((255, 255, 255))
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()


if __name__ == '__main__':
    main()
