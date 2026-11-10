"""
Loads 19 Liverpool goals from 2019, tracked from TV footage by Last Row.
https://github.com/Friends-of-Tracking-Data-FoTD/Last-Row

This is the "real TV" test. The ghost model is trained on a different
data source, so these goals are completely new to it.

Things to know about this data (from its own README):
  - It was made for animations, so it is less exact than club tracking.
  - Not every player is in every clip. Players near the ball always are.

One more thing we found: some clips have a few players who do not move
at all for the whole clip. Those are stand-ins for players who were off
screen, not real tracking. We leave them out.
"""

import numpy as np
import pandas as pd

from .clips import Clip
from .pitch import LENGTH, WIDTH, HALF_LENGTH

FPS = 20


def find_shot_frame(ball, attackers):
    """
    The shot is the last moment an attacker is touching the ball.
    We call it a touch when the ball is within 1.5 m of an attacker.
    """
    distance = np.linalg.norm(attackers - ball[:, None, :], axis=-1)
    with np.errstate(invalid="ignore"):
        touching = np.nanmin(distance, axis=1) < 1.5
    frames = np.where(touching)[0]
    return int(frames[-1]) if len(frames) else len(ball) - 1


def drop_frozen(players):
    """Remove stand-in players: anyone who covers less than half a meter in the whole clip."""
    covered = np.linalg.norm(np.nanmax(players, axis=0) - np.nanmin(players, axis=0), axis=-1)
    return players[:, covered >= 0.5]


def load_goals(csv_path):
    """Returns a list of Clips, one per goal."""
    table = pd.read_csv(csv_path, index_col=0)
    clips = []
    for play, rows in table.groupby("play", sort=True):
        frames = int(rows["frame"].max()) + 1

        def positions(part):
            ids = sorted(part["player"].unique())
            out = np.full((frames, len(ids), 2), np.nan)
            for i, player in enumerate(ids):
                one = part[part["player"] == player]
                # The file uses 0..100 on both sides. Turn that into meters from the middle.
                out[one["frame"].to_numpy(), i, 0] = (one["x"].to_numpy() / 100 - 0.5) * LENGTH
                out[one["frame"].to_numpy(), i, 1] = (one["y"].to_numpy() / 100 - 0.5) * WIDTH
            return out

        ball = positions(rows[rows["player"] == 0])[:, 0]
        attackers = drop_frozen(positions(rows[rows["team"] == "attack"]))
        defenders = drop_frozen(positions(rows[rows["team"] == "defense"]))

        # Turn the clip around if the goal is scored on the left.
        if ball[-1, 0] < 0:
            ball, attackers, defenders = -ball, -attackers, -defenders

        # The defending goalkeeper is the defender who stays closest to his goal.
        to_goal = np.nanmean(np.hypot(defenders[..., 0] - HALF_LENGTH, defenders[..., 1]), axis=0)
        def_keeper = int(np.argmin(to_goal)) if to_goal.min() < 16 else -1
        # The attacking goalkeeper is usually off screen. If someone sits deep in his own half, that is him.
        own_goal = np.nanmean(np.hypot(attackers[..., 0] + HALF_LENGTH, attackers[..., 1]), axis=0)
        att_keeper = int(np.argmin(own_goal)) if own_goal.min() < 20 else -1

        # "Liverpool [3] - 0 Bournemouth" means Liverpool scored. The team next to the brackets attacks.
        home, away = [side.strip() for side in play.split(" - ")]
        scorer_is_home = "[" in home
        strip = lambda side: " ".join(w for w in side.split() if not w.strip("[]").isdigit())
        att_team, def_team = (strip(home), strip(away)) if scorer_is_home else (strip(away), strip(home))

        clips.append(Clip(play, "Last Row (tracked from TV)", float(FPS), ball.astype(np.float32),
                          attackers.astype(np.float32), defenders.astype(np.float32),
                          att_keeper, def_keeper, find_shot_frame(ball, attackers),
                          att_team=att_team, def_team=def_team))
    return clips
