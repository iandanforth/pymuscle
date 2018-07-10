import sys
import gym
import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
from pymuscle import PotvinMuscle as Muscle


class PymunkArmEnv(gym.Env):
    """
    Single joint arm with opposing muscles.
    """

    def __init__(self):
        # Set up our 2D physics simulation
        self._init_sim()

        # Add a simulated arm consisting of:
        #  - bones (rigid bodies)
        #  - muscle bodies (damped spring constraints)
        self.brach, self.tricep = self._add_arm()

        # Instantiate the PyMuscles
        apply_fatigue = False
        brach_motor_unit_count = 100
        self.brach_muscle = Muscle(brach_motor_unit_count, apply_fatigue)
        tricep_motor_unit_count = 100
        self.tricep_muscle = Muscle(tricep_motor_unit_count, apply_fatigue)

        self.frames = 0

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

        # Scale input to match the expected range of the muscle sim
        input_array = np.array(input_array) * 67.0

        if debug:
            print(input_array)

        # Advance the simulation
        self.space.step(step_size)
        self.frames += 1

        # Advance muscle sim and sync with physics sim
        brach_output = self.brach_muscle.step(input_array[0], step_size)
        tricep_output = self.tricep_muscle.step(input_array[1], step_size)

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

    def reset(self):
        if self.space:
            del self.space
        self._init_sim()
