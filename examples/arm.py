import sys
import numpy as np
import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
import pymunk
import pymunk.pygame_util


def main():
    # Set up our 2D physics simulation
    screen, space, clock = init_sim()

    # Add a simulated arm consisting of two rigid bodies (bones) and
    # two damped springs (muscle bodies)
    brach, tricep = add_arm(screen, space)

    # Render Loop
    debug = True
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    draw_options.flags = 1  # Disable constraint drawing
    if debug:
        draw_options.flags = 3  # Enable constraint drawing
    frames = 0
    stiffness_delta = 50
    while True:
        handle_keys()

        # Advance the simulation
        space.step(1 / 50.0)

        # Redraw all objects
        screen.fill((255, 255, 255))
        space.debug_draw(draw_options)
        pygame.display.flip()

        clock.tick(50)
        frames += 1

        # Vary world conditions
        if frames % 50 == 0:
            if debug:
                print("Tricep spring stiffness: ", tricep.stiffness)
            tricep.stiffness += stiffness_delta

        if frames % 1000 == 0:
            stiffness_delta *= -1


def handle_keys():
    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit(0)
        elif event.type == KEYDOWN and event.key == K_ESCAPE:
            sys.exit(0)


def init_sim():
    pygame.init()
    screen_width = screen_height = 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Curl Sim")
    space = pymunk.Space()
    space.gravity = (0.0, -900.0)
    clock = pygame.time.Clock()

    return screen, space, clock


def add_arm(screen, space):
    config = {
        "arm_center": (screen.get_width() / 2, screen.get_height() / 2),
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

    space.add(upper_arm_body)
    space.add(upper_arm_line)

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

    space.add(lower_arm_body)
    space.add(lower_arm_line)

    # Pivot (Elbow)
    elbow_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    elbow_body.position = config["arm_center"]
    elbow_joint = pymunk.PivotJoint(elbow_body, lower_arm_body, config["arm_center"])
    space.add(elbow_joint)

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
    space.add(brach_spring)

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
    space.add(tricep_spring)

    # Elbow stop (prevent under/over extension)
    elbow_stop_point = pymunk.Circle(
        upper_arm_body,
        radius=5,
        offset=(-elbow_extension_length, -3)
    )
    elbow_stop_point.friction = 1.0
    space.add(elbow_stop_point)

    return brach_spring, tricep_spring


if __name__ == '__main__':
    main()
