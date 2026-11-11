"""
Loads 20 A-League matches (2024/25 season) shared by SkillCorner.
https://github.com/SkillCorner/opendata

This is the important dataset for this project, because SkillCorner
tracks players from the normal TV broadcast. So this is what real
"tracking from TV footage" looks like:

  - 10 frames per second.
  - The camera only shows part of the pitch. On average about 14 of the
    22 players are on screen. For the rest, SkillCorner fills in a guess
    and marks it "is_detected: false".
  - During replays and close-ups nobody is on screen at all.

We keep the guesses in the position table (so speeds stay smooth) but
mark them as not seen. The rest of the project only trusts seen players.

The matches also come with a list of events. We use it for two things:
who has the ball (from the "phases of play" file) and when shots and
goals happen (from the "dynamic events" file).
"""

import json
import os

import numpy as np
import pandas as pd

from .data_metrica import Match

FPS = 10


def match_ids(folder):
    """The ids of all matches in the SkillCorner 'data' folder, oldest first."""
    with open(os.path.join(folder, "matches.json")) as f:
        listing = json.load(f)
    return [m["id"] for m in sorted(listing, key=lambda m: m["date_time"])]


def load_match(folder, match_id, cache_folder=None):
    """
    Load one match. folder is the SkillCorner 'data' folder.

    Reading the big tracking file takes a few seconds, so the result can
    be saved in cache_folder and read back quickly next time.
    """
    here = os.path.join(folder, "matches", str(match_id))
    with open(os.path.join(here, f"{match_id}_match.json")) as f:
        info = json.load(f)

    team_ids = [info["home_team"]["id"], info["away_team"]["id"]]
    teams = (info["home_team"]["short_name"], info["away_team"]["short_name"])

    # Give every player who took part a fixed slot inside his team.
    slot, numbers_list, keepers = {}, [[], []], [[], []]
    for player in info["players"]:
        if player["playing_time"]["total"] is None:
            continue  # he was on the bench the whole match
        team = team_ids.index(player["team_id"])
        slot[player["id"]] = (team, len(numbers_list[team]))
        numbers_list[team].append(int(player["number"] or 0))
        if player["player_role"]["acronym"] == "GK":
            keepers[team].append(len(numbers_list[team]) - 1)
    slots = max(len(numbers_list[0]), len(numbers_list[1]))
    numbers = np.zeros((2, slots), dtype=np.int64)
    for team in (0, 1):
        numbers[team, :len(numbers_list[team])] = numbers_list[team]

    tracking = _read_tracking(os.path.join(here, f"{match_id}_tracking_extrapolated.jsonl"), slot, slots,
                              os.path.join(cache_folder, f"{match_id}.npz") if cache_folder else None)
    xy, seen, ball = tracking["xy"], tracking["seen"], tracking["ball"]
    frames = len(ball)

    period = np.zeros(frames, dtype=np.int8)
    for p in info["match_periods"]:
        if p["period"] in (1, 2):
            period[p["start_frame"]:p["end_frame"] + 1] = p["period"]

    # Who has the ball: the "phases of play" file lists stretches of live
    # play and the team in possession during each one.
    phases = pd.read_csv(os.path.join(here, f"{match_id}_phases_of_play.csv"))
    possession = np.full(frames, -1, dtype=np.int8)
    for start, end, team_id in zip(phases["frame_start"], phases["frame_end"], phases["team_in_possession_id"]):
        possession[int(start):int(end)] = team_ids.index(int(team_id))

    # We treat the ball as "out of play" whenever nobody is in a phase of
    # play, the same way the Metrica matches do it.
    live = (possession >= 0) & (period > 0) & ~np.isnan(ball[:, 0])
    ball = ball.copy()
    ball[~live] = np.nan
    possession[~live] = -1

    direction, keeper = {}, {}
    for p in (1, 2):
        home_goes_right = info["home_team_side"][p - 1] == "left_to_right"
        direction[(p, 0)] = 1 if home_goes_right else -1
        direction[(p, 1)] = -1 if home_goes_right else 1
        in_period = period == p
        for team in (0, 1):
            # A team can change goalkeeper. Take the one who played most of this half.
            time_on = [np.sum(~np.isnan(xy[in_period, team, k, 0])) for k in keepers[team]]
            keeper[(p, team)] = keepers[team][int(np.argmax(time_on))] if keepers[team] else -1

    shots = _read_shots(os.path.join(here, f"{match_id}_dynamic_events.csv"), team_ids, possession)

    score = f"{info['home_team_score']}-{info['away_team_score']}"
    label = f"{teams[0]} {score} {teams[1]}"
    return Match(f"A-League {info['date_time'][:10]} {label}", FPS, xy, ball, period, possession, direction, keeper,
                 shots, seen=seen, numbers=numbers, teams=teams, source="skillcorner", label=label)


def _read_tracking(path, slot, slots, cache_path):
    """Read the tracking file into arrays. One line of the file is one frame."""
    if cache_path and os.path.exists(cache_path):
        return dict(np.load(cache_path))

    with open(path) as f:
        lines = f.readlines()
    frames = len(lines)
    xy = np.full((frames, 2, slots, 2), np.nan, dtype=np.float32)
    seen = np.zeros((frames, 2, slots), dtype=bool)
    ball = np.full((frames, 2), np.nan, dtype=np.float32)

    for line in lines:
        row = json.loads(line)
        frame = row["frame"]
        for player in row["player_data"]:
            where = slot.get(player["player_id"])
            if where is None:
                continue
            xy[frame, where[0], where[1]] = player["x"], player["y"]
            seen[frame, where[0], where[1]] = bool(player["is_detected"])
        if row["ball_data"]["x"] is not None:
            ball[frame] = row["ball_data"]["x"], row["ball_data"]["y"]

    out = {"xy": xy, "seen": seen, "ball": ball}
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, **out)
    return out


def _read_shots(path, team_ids, possession):
    """Find every shot. A shot is a goal when play stops for a goal right after it."""
    columns = ["event_type", "end_type", "frame_end", "team_id", "game_interruption_after"]
    events = pd.read_csv(path, usecols=columns, low_memory=False)
    events = events[(events["event_type"] == "player_possession") & (events["end_type"] == "shot")]
    shots = []
    for frame, team_id, after in zip(events["frame_end"], events["team_id"], events["game_interruption_after"]):
        team = team_ids.index(int(team_id))
        frame = int(frame)
        # The shot is the last frame of the player's possession. That can be
        # one frame past the end of the phase of play, so step back a little
        # until we are inside it.
        for _ in range(5):
            if frame < len(possession) and possession[frame] == team:
                break
            frame -= 1
        shots.append({"frame": frame, "team": team, "goal": after == "goal_for", "penalty": False})
    return shots


def load_all(folder, cache_folder=None):
    """Load every match. folder is the 'data' folder of the SkillCorner repo."""
    return [load_match(folder, match_id, cache_folder) for match_id in match_ids(folder)]
