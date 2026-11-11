"""
Checks on the real datasets. They only run if you have downloaded the
data into the data folder (see the README). Otherwise they are skipped.
"""

import os

import numpy as np
import pytest

from ghostxi.pitch import HALF_LENGTH

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
METRICA = os.path.join(DATA, "metrica", "data")
SKILLCORNER = os.path.join(DATA, "skillcorner", "data")
LASTROW = os.path.join(DATA, "lastrow", "datasets", "positional_data", "liverpool_2019.csv")


@pytest.mark.skipif(not os.path.exists(METRICA), reason="Metrica data not downloaded")
def test_every_goal_ends_in_the_goal_the_team_attacks():
    from ghostxi.data_metrica import load_all
    goals = 0
    for match in load_all(METRICA):
        assert match.xy.shape[1] == 2
        for shot in match.shots:
            if not shot["goal"]:
                continue
            goals += 1
            frame = shot["frame"]
            sign = match.direction[(int(match.period[frame]), shot["team"])]
            after = match.ball[frame:frame + 60]
            after = after[~np.isnan(after[:, 0])]
            assert after[-1, 0] * sign > HALF_LENGTH - 2      # the ball ends up at the goal line
            assert match.possession[frame] == shot["team"]
    assert goals == 10


@pytest.mark.skipif(not os.path.exists(LASTROW), reason="Last Row data not downloaded")
def test_tv_goals_load_cleanly():
    from ghostxi.data_lastrow import load_goals
    clips = load_goals(LASTROW)
    assert len(clips) == 19
    for clip in clips:
        assert clip.ball[-1, 0] > HALF_LENGTH - 2             # every clip ends with the ball in the goal
        assert clip.attackers.shape[1] <= 11 and clip.defenders.shape[1] <= 11
        assert not np.isnan(clip.defenders).any()
        assert 0 < clip.shot_frame < len(clip.ball)
        # Nobody is frozen in place (those stand-ins were dropped).
        covered = np.linalg.norm(clip.defenders.max(0) - clip.defenders.min(0), axis=-1)
        assert covered.min() >= 0.5


@pytest.mark.skipif(not os.path.exists(SKILLCORNER), reason="SkillCorner data not downloaded")
def test_tv_tracked_matches_load_cleanly():
    from ghostxi.data_skillcorner import load_match, match_ids
    ids = match_ids(SKILLCORNER)
    assert len(ids) == 20
    for match_id in (ids[0], ids[-1]):
        match = load_match(SKILLCORNER, match_id, cache_folder=os.path.join(DATA, "cache"))
        assert match.fps == 10 and match.source == "skillcorner"
        live = match.possession >= 0
        assert live.sum() > 20 * 60 * match.fps                     # more than 20 minutes of live play
        on_screen = match.seen[live].sum(axis=(1, 2)).mean()
        assert 8 < on_screen < 20                                   # a TV camera shows about 14 of 22 players
        assert not match.seen[np.isnan(match.xy[..., 0])].any()     # nobody is "seen" without a position
        assert len(match.shots) >= 10
        for shot in match.shots:
            frame = shot["frame"]
            sign = match.direction[(int(match.period[frame]), shot["team"])]
            assert match.possession[frame] == shot["team"]
            assert match.ball[frame, 0] * sign > 0                  # shots are taken in the half the team attacks
        for period in (1, 2):
            for team in (0, 1):
                keeper = match.keeper[(period, team)]
                x = np.nanmean(match.xy[(match.period == period) & live, team, keeper, 0])
                assert x * match.direction[(period, team)] < -25    # the goalkeeper stays near his own goal
