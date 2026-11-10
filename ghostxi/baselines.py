"""
Simple ways to guess where defenders go, with no machine learning.

A model is only worth having if it beats simple guesses, so we measure
these too. All of them take clips (see windows.py) and return paths
shaped like the ghost paths: (clips, 25, 11, 2).
"""

import numpy as np

from .data import STEP


def stand_still(clips):
    """Everyone freezes where they were at the start."""
    start = clips["defenders"][:, 1:2]
    return np.repeat(start, clips["defenders"].shape[1] - 1, axis=1)


def keep_running(clips):
    """Everyone keeps moving in a straight line at their starting speed."""
    start = clips["defenders"][:, 1]
    speed = (clips["defenders"][:, 1] - clips["defenders"][:, 0]) / STEP
    steps = np.arange(clips["defenders"].shape[1] - 1)[None, :, None, None]
    return start[:, None] + speed[:, None] * steps * STEP


def fit_follow_ball(clips):
    """
    Learn how much defenders shift when the ball moves.
    If the ball moves 10 m sideways, the defence slides maybe 4 m the same way.
    We find the best fraction for length and for width from training data.
    """
    ball_move = clips["ball"][:, 1:] - clips["ball"][:, 1:2]
    player_move = clips["defenders"][:, 1:] - clips["defenders"][:, 1:2]
    on = clips["defender_on"] & ~clips["keeper"]
    fractions = []
    for axis in (0, 1):
        b = np.repeat(ball_move[:, :, None, axis], 11, axis=2)[:, :, :][np.broadcast_to(on[:, None, :], player_move.shape[:3])]
        p = player_move[..., axis][np.broadcast_to(on[:, None, :], player_move.shape[:3])]
        fractions.append(float((b * p).sum() / (b * b).sum()))
    return np.array(fractions)


def follow_ball(clips, fractions):
    """Everyone shifts by a fixed fraction of how far the ball has moved."""
    ball_move = clips["ball"][:, 1:] - clips["ball"][:, 1:2]
    return clips["defenders"][:, 1:2] + ball_move[:, :, None, :] * fractions


def average_miss(paths, clips, at_seconds=None, fps=5):
    """
    How far off are the paths, in metres, averaged over outfield defenders?
    at_seconds=None gives the average over the whole 5 seconds.
    at_seconds=3 gives the miss exactly 3 seconds in.
    """
    real = clips["defenders"][:, 1:]
    miss = np.linalg.norm(paths - real, axis=-1)            # (clips, steps, 11)
    count = clips["defender_on"] & ~clips["keeper"]         # outfield players only
    if at_seconds is not None:
        miss = miss[:, at_seconds * fps:at_seconds * fps + 1]
    weights = np.broadcast_to(count[:, None, :], miss.shape)
    return float((miss * weights).sum() / weights.sum())
