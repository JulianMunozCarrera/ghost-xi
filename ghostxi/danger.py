"""
Measures how dangerous a moment is for the defending team.

The idea in plain words: look at the space in front of the goal. For
every spot, ask "who would get there first, an attacker or a defender?"
Spots the attackers would win are dangerous. Spots with a wide open
view of the goal count more than spots at a tight angle.

    danger = share of the shooting space the attackers would win
             (0 = defenders own all of it, 1 = attackers own all of it)

The "who gets there first" part is a simplified version of the pitch
control model from William Spearman's paper "Beyond Expected Goals"
(2018). Every player reacts for 0.7 seconds, then runs at 5 m/s. Those
two numbers are the usual ones for this model.
"""

import numpy as np

from .pitch import HALF_LENGTH, GOAL_WIDTH

REACTION = 0.7      # seconds before a player can change what he is doing
TOP_SPEED = 5.0     # meters per second
FUZZ = 0.45         # how unsure we are about arrival times, in seconds
REACH = 24.0        # we look at spots up to this far from the goal, in meters


def _zone():
    """A grid of spots (1 m apart) in front of the goal, and how much each one counts."""
    xs = np.arange(HALF_LENGTH - REACH + 0.5, HALF_LENGTH, 1.0)
    ys = np.arange(-REACH + 0.5, REACH, 1.0)
    grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    from_goal = np.hypot(HALF_LENGTH - grid[:, 0], grid[:, 1])
    grid = grid[from_goal <= REACH]

    # How much of the goal can you see from each spot? This is the angle
    # between the two posts as seen from the spot. Wide angle = good chance.
    dx = HALF_LENGTH - grid[:, 0]
    y = grid[:, 1]
    angle = np.arctan2(GOAL_WIDTH * dx, dx ** 2 + y ** 2 - (GOAL_WIDTH / 2) ** 2)
    return grid, angle


ZONE, ZONE_WEIGHT = _zone()


def arrival_times(spots, players, speeds):
    """Seconds each player needs to reach each spot. Shape (players, spots)."""
    after_reaction = players + speeds * REACTION  # he keeps drifting while he reacts
    distance = np.linalg.norm(after_reaction[:, None, :] - spots[None, :, :], axis=-1)
    return REACTION + distance / TOP_SPEED


def attack_control(spots, attackers, att_speeds, defenders, def_speeds):
    """For each spot: the chance (0 to 1) that an attacker gets there first."""
    attackers, att_speeds = _drop_missing(attackers, att_speeds)
    defenders, def_speeds = _drop_missing(defenders, def_speeds)
    if len(attackers) == 0:
        return np.zeros(len(spots))
    if len(defenders) == 0:
        return np.ones(len(spots))
    first_attacker = arrival_times(spots, attackers, att_speeds).min(axis=0)
    first_defender = arrival_times(spots, defenders, def_speeds).min(axis=0)
    return 1.0 / (1.0 + np.exp(-(first_defender - first_attacker) / FUZZ))


def danger(attackers, att_speeds, defenders, def_speeds):
    """The danger score for one moment, between 0 and 1."""
    control = attack_control(ZONE, attackers, att_speeds, defenders, def_speeds)
    return float((control * ZONE_WEIGHT).sum() / ZONE_WEIGHT.sum())


def space(ball, defenders, keeper=-1):
    """Distance in meters from the ball to the nearest outfield defender."""
    keep = [i for i in range(len(defenders)) if i != keeper and not np.isnan(defenders[i]).any()]
    if not keep:
        return float("nan")
    return float(np.linalg.norm(defenders[keep] - ball, axis=-1).min())


def _drop_missing(players, speeds):
    ok = ~np.isnan(players).any(axis=-1)
    return np.asarray(players)[ok], np.asarray(speeds)[ok]
