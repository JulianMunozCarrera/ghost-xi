"""
Measures how dangerous a situation is for the defending team.

The idea is called "pitch control". For any spot on the pitch we ask:
if the ball dropped there right now, which team would get to it first?

We look at the area close to the goal (the danger zone) and work out
how much of it the attackers would win. A good defensive shape gives
the attackers very little of it.

This is a simple version of the model from Spearman (2018) and Laurie
Shaw's Friends of Tracking lessons.
"""

import numpy as np

from .pitch import HALF_LENGTH, HALF_WIDTH, OWN_GOAL

REACTION = 0.7        # seconds before a player can change what they are doing
TOP_SPEED = 5.0       # metres per second, a normal sprint
FUZZ = 0.45           # how unsure we are about arrival times, in seconds
DANGER_RADIUS = 25.0  # the danger zone is everything this close to the goal


def danger_zone(spacing=2.0):
    """
    Grid points inside the pitch and within 25 m of the goal being defended.
    Also returns a weight for each point. Closer to goal counts more.
    """
    xs = np.arange(-HALF_LENGTH + spacing / 2, -HALF_LENGTH + DANGER_RADIUS, spacing)
    ys = np.arange(-HALF_WIDTH + spacing / 2, HALF_WIDTH, spacing)
    grid = np.array([(x, y) for x in xs for y in ys])
    distance = np.linalg.norm(grid - OWN_GOAL, axis=1)
    inside = distance < DANGER_RADIUS
    return grid[inside], 1.0 - distance[inside] / DANGER_RADIUS


def time_to_reach(players, player_speeds, spots):
    """
    Seconds for each player to reach each spot. Shape (players, spots).
    For the reaction time they keep moving as they were. Then they
    sprint straight to the spot.
    """
    after_reaction = players + player_speeds * REACTION
    distance = np.linalg.norm(spots[None, :, :] - after_reaction[:, None, :], axis=-1)
    return REACTION + distance / TOP_SPEED


def attack_share(defenders, defender_speeds, attackers, attacker_speeds, spots=None, weights=None):
    """
    How much of the danger zone would the attackers win? 0 = none, 1 = all.
    Pass only the players who are on the pitch.
    """
    if spots is None:
        spots, weights = danger_zone()
    first_defender = time_to_reach(defenders, defender_speeds, spots).min(axis=0)
    first_attacker = time_to_reach(attackers, attacker_speeds, spots).min(axis=0)
    # If the attacker arrives well before the defender this is close to 1.
    attacker_wins = 1.0 / (1.0 + np.exp(-(first_defender - first_attacker) / FUZZ))
    return float((attacker_wins * weights).sum() / weights.sum())
