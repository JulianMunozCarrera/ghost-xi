"""
The size of the pitch and the lines painted on it.

We measure everything in metres, with (0, 0) at the centre spot.
x runs along the length of the pitch (goal to goal).
y runs across the width (touchline to touchline).
"""

import numpy as np

LENGTH = 105.0   # goal line to goal line
WIDTH = 68.0     # touchline to touchline
HALF_LENGTH = LENGTH / 2
HALF_WIDTH = WIDTH / 2

GOAL_WIDTH = 7.32
PENALTY_BOX_DEPTH = 16.5
PENALTY_BOX_WIDTH = 40.32
SIX_YARD_DEPTH = 5.5
SIX_YARD_WIDTH = 18.32
CENTRE_CIRCLE = 9.15
PENALTY_SPOT = 11.0

# In this project the defending team always defends the LEFT goal.
OWN_GOAL = np.array([-HALF_LENGTH, 0.0])


def pitch_lines():
    """
    Every white line on the pitch as a list of point paths.
    Each path is a list of (x, y) points to join up.
    Used to draw the pitch and to find the camera position from video.
    """
    L, W = HALF_LENGTH, HALF_WIDTH
    lines = [
        [(-L, -W), (L, -W), (L, W), (-L, W), (-L, -W)],   # outside edge
        [(0, -W), (0, W)],                                # halfway line
    ]
    for side in (-1, 1):  # left goal, then right goal
        goal_x = side * L
        for depth, width in ((PENALTY_BOX_DEPTH, PENALTY_BOX_WIDTH), (SIX_YARD_DEPTH, SIX_YARD_WIDTH)):
            inner_x = goal_x - side * depth
            lines.append([(goal_x, -width / 2), (inner_x, -width / 2), (inner_x, width / 2), (goal_x, width / 2)])

        # The curved bit on the edge of the penalty box (the "D").
        spot_x = goal_x - side * PENALTY_SPOT
        reach = np.arccos((PENALTY_BOX_DEPTH - PENALTY_SPOT) / CENTRE_CIRCLE)
        angles = np.linspace(-reach, reach, 20)
        lines.append([(spot_x - side * CENTRE_CIRCLE * np.cos(a), CENTRE_CIRCLE * np.sin(a)) for a in angles])

    angles = np.linspace(0, 2 * np.pi, 60)
    lines.append([(CENTRE_CIRCLE * np.cos(a), CENTRE_CIRCLE * np.sin(a)) for a in angles])
    return lines
