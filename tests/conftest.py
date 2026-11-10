import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from ghostxi.clips import Clip
from ghostxi.data_metrica import Match


def _fake_match(tv):
    """
    A small made-up match so the tests run without downloading anything.
    Players wander around randomly. One slot is an unused substitute.
    With tv=True, players far from the ball are "off screen", like TV tracking.
    """
    rng = np.random.default_rng(1)
    frames, slots = 1500, 12
    start = rng.uniform([-40, -25], [40, 25], (2, slots, 2))
    steps = rng.normal(0, 0.12, (frames, 2, slots, 2)).cumsum(axis=0)
    xy = (start + steps).astype(np.float32)
    xy[:, :, 11] = np.nan                    # the substitute never comes on
    xy[:, 0, 0] = [-48, 0]                   # home keeper stays near his goal
    xy[:, 1, 0] = [48, 0]                    # away keeper too
    ball = (rng.normal(0, 0.4, (frames, 2)).cumsum(axis=0)).astype(np.float32)
    ball[600:650] = np.nan                   # the ball goes out of play for two seconds
    period = np.ones(frames, dtype=np.int8)
    possession = np.where(np.arange(frames) // 250 % 2 == 0, 0, 1).astype(np.int8)
    possession[np.isnan(ball[:, 0])] = -1
    direction = {(1, 0): 1, (1, 1): -1, (2, 0): -1, (2, 1): 1}
    keeper = {(1, 0): 0, (1, 1): 0, (2, 0): 0, (2, 1): 0}
    seen = None
    if tv:
        # The camera shows 30 m either side of the ball (and nothing while the ball is out).
        seen = np.abs(xy[..., 0] - np.nan_to_num(ball[:, None, None, 0], nan=999.0)) < 30
    return Match("fake match", 25, xy, ball, period, possession, direction, keeper, [], seen=seen)


@pytest.fixture
def fake_match():
    return _fake_match(tv=False)


@pytest.fixture
def fake_tv_match():
    return _fake_match(tv=True)


@pytest.fixture
def fake_clip():
    """Ten seconds of made-up play: 6 attackers and 7 defenders drifting toward the goal."""
    rng = np.random.default_rng(2)
    frames, fps = 200, 20.0
    time = np.arange(frames)[:, None, None] / fps
    attackers = rng.uniform([0, -25], [25, 25], (1, 6, 2)) + time * [2.0, 0.0] + rng.normal(0, 0.02, (frames, 6, 2)).cumsum(0)
    defenders = rng.uniform([20, -25], [40, 25], (1, 7, 2)) + time * [1.2, 0.0] + rng.normal(0, 0.02, (frames, 7, 2)).cumsum(0)
    defenders[:, 6] = [50.0, 0.0] + rng.normal(0, 0.01, (frames, 2)).cumsum(0)   # the goalkeeper
    ball = attackers[:, 0] + [0.5, 0.0]
    return Clip("fake goal", "made up", fps, ball, attackers, defenders, -1, 6, 180)


@pytest.fixture
def fake_tv_clip(fake_clip):
    """The same clip, but two defenders and one attacker are off screen for part of it."""
    defenders = fake_clip.defenders.copy()
    attackers = fake_clip.attackers.copy()
    defenders[:120, 2] = np.nan      # defender 2 comes into the picture late (no "then")
    defenders[150:, 4] = np.nan      # defender 4 leaves the picture before the shot
    attackers[:100, 3] = np.nan      # attacker 3 comes into the picture late
    return Clip("fake tv goal", "made up", fake_clip.fps, fake_clip.ball, attackers, defenders, -1, 6, 180)
