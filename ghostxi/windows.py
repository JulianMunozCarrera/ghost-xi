"""
Cuts defending spells into short clips of equal length.

The model trains and gets tested on these clips. Each clip is 5 seconds
of play, plus one extra frame at the start to measure starting speeds.
"""

import numpy as np

from .data import FPS, defending_spells

SECONDS = 5
FRAMES = SECONDS * FPS + 2   # 25 steps to predict + start frame + 1 frame before it for speed


def make_windows(matches, every=2):
    """
    Slide a 5 second window over every defending spell.
    every=2 means a new clip starts every 2 frames (0.4 seconds).
    Returns a dict of arrays, one row per clip.
    """
    clips = {k: [] for k in ("defenders", "attackers", "ball", "defender_on", "attacker_on", "keeper")}
    for match in matches:
        for spell in defending_spells(match):
            for start in range(0, len(spell.ball) - FRAMES + 1, every):
                cut = slice(start, start + FRAMES)
                clips["defenders"].append(spell.defenders[cut])
                clips["attackers"].append(spell.attackers[cut])
                clips["ball"].append(spell.ball[cut])
                clips["defender_on"].append(spell.defender_on)
                clips["attacker_on"].append(spell.attacker_on)
                clips["keeper"].append(spell.keeper)
    return {k: np.stack(v) for k, v in clips.items()}


def flip_sides(clips):
    """
    Mirror every clip across the long axis of the pitch (left wing becomes
    right wing). Football is symmetric this way, so this doubles our data.
    """
    mirrored = dict(clips)
    for name in ("defenders", "attackers", "ball"):
        copy = clips[name].copy()
        copy[..., 1] *= -1
        mirrored[name] = copy
    return mirrored
