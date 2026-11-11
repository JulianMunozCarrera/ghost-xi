"""
The size of the pitch, and a helper to turn positions into speeds.

We measure everything in meters, with (0, 0) at the center spot.
x runs along the length of the pitch (goal to goal).
y runs across the width (touchline to touchline).
"""

import numpy as np

LENGTH = 105.0   # goal line to goal line
WIDTH = 68.0     # touchline to touchline
HALF_LENGTH = LENGTH / 2
HALF_WIDTH = WIDTH / 2
GOAL_WIDTH = 7.32

MAX_PLAYERS = 11   # players per team on the pitch


def speed_from_positions(xy, fps, reach=None, top_speed=12.0):
    """
    How fast something moves at every frame, in meters per second.

    xy has frames along its first axis and (x, y) along its last.
    For each frame we compare the position a few frames ahead with the
    position a few frames back. 'reach' is how many frames that is.

    Players can be missing (NaN). If one side is missing we use the
    other side. If we cannot tell, the speed is 0. Nobody runs faster
    than top_speed, so anything above that is a tracking glitch and
    gets capped.

    The website does the exact same steps in JavaScript (web/engine.js).
    """
    xy = np.asarray(xy, dtype=np.float32)
    if reach is None:
        reach = max(1, round(0.12 * fps))
    ahead = np.full_like(xy, np.nan)
    back = np.full_like(xy, np.nan)
    ahead[:-reach] = xy[reach:]
    back[reach:] = xy[:-reach]

    here = ~np.isnan(xy[..., :1])
    has_ahead = ~np.isnan(ahead[..., :1])
    has_back = ~np.isnan(back[..., :1])
    one_side = np.float32(reach / fps)   # seconds between this frame and 'reach' frames away

    with np.errstate(invalid="ignore"):
        both = (ahead - back) / (2 * one_side)
        forward = (ahead - xy) / one_side
        backward = (xy - back) / one_side
    speed = np.where(has_ahead & has_back, both,
                     np.where(has_ahead & here, forward,
                              np.where(has_back & here, backward, 0.0)))
    speed = np.nan_to_num(speed).astype(np.float32)

    size = np.linalg.norm(speed, axis=-1, keepdims=True)
    too_fast = size > top_speed
    return np.where(too_fast, speed * top_speed / np.maximum(size, 1e-9), speed).astype(np.float32)
