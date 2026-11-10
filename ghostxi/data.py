"""
Reads real match tracking data and gets it ready for the model.

Tracking data = the (x, y) position of every player and the ball, many
times per second, for a whole match. We use the three open sample
matches from Metrica Sports.

Two jobs happen here:
  1. Read the files (two different formats) into one common shape.
  2. Cut each match into "defending spells": stretches where one team
     has the ball and the other team is defending.
"""

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .pitch import LENGTH, WIDTH

RAW_FPS = 25      # the files have 25 positions per second
FPS = 5           # we keep 5 per second. Plenty for players, and 5x less data.
STEP = 1.0 / FPS  # seconds between two of our frames
MAX_PLAYERS = 11

# Event types that tell us who has the ball.
HAS_BALL_EVENTS = {"PASS", "CARRY", "SHOT", "SET PIECE", "RECOVERY"}


@dataclass
class Match:
    name: str
    home: np.ndarray        # (frames, players, 2) positions in metres. NaN = not on the pitch.
    away: np.ndarray
    ball: np.ndarray        # (frames, 2). NaN = ball is out of play.
    period: np.ndarray      # (frames,) 1 = first half, 2 = second half
    has_ball: np.ndarray    # (frames,) 0 = nobody, 1 = home, 2 = away
    raw_frame: np.ndarray   # (frames,) the frame number in the original file
    events: list = field(default_factory=list)

    def team(self, side):
        return self.home if side == "home" else self.away


def _to_metres(xy):
    """The files use 0 to 1 for both axes. Turn that into metres from the centre spot."""
    out = np.empty_like(xy, dtype=np.float32)
    out[..., 0] = (xy[..., 0] - 0.5) * LENGTH
    out[..., 1] = (0.5 - xy[..., 1]) * WIDTH
    return out


def _smooth_and_thin(xy):
    """
    Average each position with its neighbours to remove camera jitter,
    then keep every 5th frame (25 per second becomes 5 per second).
    """
    kernel = np.ones(5) / 5
    flat = xy.reshape(len(xy), -1)
    smooth = np.stack([np.convolve(flat[:, i], kernel, mode="same") for i in range(flat.shape[1])], axis=1)
    return smooth.reshape(xy.shape)[::RAW_FPS // FPS].astype(np.float32)


def _who_has_ball(events, raw_frames):
    """For every frame, which team had the ball last (1 = home, 2 = away)."""
    starts = np.array([e["start"] for e in events if e["type"] in HAS_BALL_EVENTS])
    teams = np.array([1 if e["team"] == "home" else 2 for e in events if e["type"] in HAS_BALL_EVENTS])
    order = np.argsort(starts, kind="stable")
    starts, teams = starts[order], teams[order]
    latest = np.searchsorted(starts, raw_frames, side="right") - 1
    return np.where(latest >= 0, teams[np.maximum(latest, 0)], 0).astype(np.int8)


def _finish(name, home, away, ball, period, raw_frame, events):
    thin = slice(None, None, RAW_FPS // FPS)
    raw_frame = raw_frame[thin]
    return Match(
        name=name,
        home=_smooth_and_thin(_to_metres(home)),
        away=_smooth_and_thin(_to_metres(away)),
        ball=_smooth_and_thin(_to_metres(ball)),
        period=period[thin].astype(np.int8),
        has_ball=_who_has_ball(events, raw_frame),
        raw_frame=raw_frame,
        events=events,
    )


def load_csv_match(folder, name):
    """Read sample games 1 and 2 (the CSV format)."""
    def read(side):
        path = os.path.join(folder, f"{name}_RawTrackingData_{side}_Team.csv")
        table = pd.read_csv(path, skiprows=3, header=None).to_numpy(dtype=np.float64)
        players = table[:, 3:-2].reshape(len(table), -1, 2)
        return table[:, 0], table[:, 1], players, table[:, -2:]

    period, raw_frame, home, ball = read("Home")
    _, _, away, _ = read("Away")

    events = []
    table = pd.read_csv(os.path.join(folder, f"{name}_RawEventsData.csv"))
    for row in table.itertuples(index=False):
        events.append({
            "team": row[0].lower(), "type": row[1], "subtype": str(row[2]) if isinstance(row[2], str) else "",
            "period": int(row[3]), "start": int(row[4]), "end": int(row[6]),
        })
    return _finish(name, home, away, ball, period, raw_frame.astype(int), events)


def load_epts_match(folder, name):
    """
    Read sample game 3 (the FIFA EPTS format).
    Each line is  frame:player;player;...:ball  and a separate file says
    which player sits in which slot. The slots change after a substitution.
    """
    meta = open(os.path.join(folder, f"{name}_metadata.xml")).read()
    team_of = dict(re.findall(r'<Player id="(\w+)" teamId="(\w+)">', meta))
    channel_player = dict(re.findall(r'<PlayerChannel channelId="x" id="(\w+)" playerId="(\w+)"/>', meta))
    home_team = re.search(r'idLocalTeam="(\w+)"', meta).group(1)
    half_time = int(re.search(r"<Name>second_half_start</Name>\s*<Value>(\d+)</Value>", meta).group(1))

    # Each block says: from frame A to frame B, the slots hold these players.
    blocks = []
    for end, start, body in re.findall(
            r'<DataFormatSpecification endFrame="(\d+)" separator=":" startFrame="(\d+)">(.*?)</DataFormatSpecification>',
            meta, flags=re.S):
        slots = [channel_player[c] for c in re.findall(r'playerChannelId="(\w+_x)"', body)]
        blocks.append((int(start), int(end), slots))

    everyone = sorted(team_of)
    home_ids = [p for p in everyone if team_of[p] == home_team]
    away_ids = [p for p in everyone if team_of[p] != home_team]
    column = {p: i for i, p in enumerate(home_ids)}
    column.update({p: i for i, p in enumerate(away_ids)})

    lines = open(os.path.join(folder, f"{name}_tracking.txt")).read().split("\n")
    lines = [line for line in lines if line]
    home = np.full((len(lines), len(home_ids), 2), np.nan)
    away = np.full((len(lines), len(away_ids), 2), np.nan)
    ball = np.full((len(lines), 2), np.nan)
    raw_frame = np.zeros(len(lines), dtype=int)
    for i, line in enumerate(lines):
        number, players, ball_text = line.split(":")
        raw_frame[i] = int(number)
        slots = next(s for start, end, s in blocks if start <= raw_frame[i] <= end)
        for player, text in zip(slots, players.split(";")):
            target = home if team_of[player] == home_team else away
            target[i, column[player]] = [float(v) for v in text.split(",")]
        ball[i] = [float(v) for v in ball_text.split(",")]
    period = np.where(raw_frame >= half_time, 2, 1)

    events = []
    for e in json.load(open(os.path.join(folder, f"{name}_events.json")))["data"]:
        subtypes = e["subtypes"] or []
        if isinstance(subtypes, dict):
            subtypes = [subtypes]
        events.append({
            "team": "home" if e["team"]["id"] == home_team else "away",
            "type": e["type"]["name"], "subtype": "-".join(s["name"] for s in subtypes),
            "period": e["period"], "start": e["start"]["frame"], "end": e["end"]["frame"],
        })
    return _finish(name, home, away, ball, period, raw_frame, events)


def load_all(data_folder):
    """Load the three Metrica sample matches."""
    matches = []
    for name in ("Sample_Game_1", "Sample_Game_2"):
        matches.append(load_csv_match(os.path.join(data_folder, name), name))
    matches.append(load_epts_match(os.path.join(data_folder, "Sample_Game_3"), "Sample_Game_3"))
    return matches


# ---------------------------------------------------------------------------
# Turning a match into defending spells
# ---------------------------------------------------------------------------

@dataclass
class Spell:
    """
    One stretch of play where one team defends.

    Everything is turned around so the defenders always protect the LEFT
    goal. That way the model never has to learn "left" and "right" twice.
    """
    match: str
    defending: str            # "home" or "away"
    start: int                # index of the first frame in the match
    defenders: np.ndarray     # (frames, 11, 2)
    attackers: np.ndarray     # (frames, 11, 2)
    ball: np.ndarray          # (frames, 2)
    defender_on: np.ndarray   # (11,) True where that slot holds a real player
    attacker_on: np.ndarray   # (11,)
    keeper: np.ndarray        # (11,) True for the goalkeeper


def defends_left(match, side, period):
    """Is this team defending the left goal in this half? We check where they stand at kick off."""
    frames = np.where(match.period == period)[0][:FPS * 2]
    return np.nanmean(match.team(side)[frames, :, 0]) < 0


def find_keeper(match, side):
    """The goalkeeper is the player who lives closest to a goal line."""
    typical_x = np.nanmean(np.abs(match.team(side)[:, :, 0]), axis=0)
    return typical_x > 35


def view_for_defence(match, defending, frames):
    """
    Get players and ball for some frames, turned so the defending team
    protects the left goal. Returns a Spell, or None if the numbers are off
    (for example a substitution happens in the middle).
    """
    attacking = "away" if defending == "home" else "home"
    period = int(match.period[frames[0]])
    turn = 1.0 if defends_left(match, defending, period) else -1.0

    def pack(side):
        xy = match.team(side)[frames]
        on = ~np.isnan(xy[:, :, 0])
        steady = np.all(on == on[0], axis=0).all()
        players = np.where(on[0])[0]
        if not steady or len(players) > MAX_PLAYERS or len(players) < 7:
            return None
        out = np.zeros((len(frames), MAX_PLAYERS, 2), dtype=np.float32)
        out[:, :len(players)] = xy[:, players] * turn
        mask = np.zeros(MAX_PLAYERS, dtype=bool)
        mask[:len(players)] = True
        keeper = np.zeros(MAX_PLAYERS, dtype=bool)
        keeper[:len(players)] = find_keeper(match, side)[players]
        return out, mask, keeper

    defence, attack = pack(defending), pack(attacking)
    if defence is None or attack is None:
        return None
    return Spell(match.name, defending, int(frames[0]), defence[0], attack[0],
                 match.ball[frames] * turn, defence[1], attack[1], defence[2])


def defending_spells(match, shortest=3.0):
    """
    Find every stretch where:
      - the ball is in play,
      - the same team keeps the ball,
      - nobody is substituted,
      - and it lasts at least 'shortest' seconds.
    """
    alive = ~np.isnan(match.ball[:, 0])
    key = np.where(alive, match.has_ball.astype(int) + 10 * match.period, -1)
    spells = []
    start = 0
    for i in range(1, len(key) + 1):
        if i == len(key) or key[i] != key[start]:
            if key[start] > 0 and match.has_ball[start] > 0 and (i - start) >= shortest * FPS:
                defending = "away" if match.has_ball[start] == 1 else "home"
                spell = view_for_defence(match, defending, np.arange(start, i))
                if spell is not None and _speeds_look_real(spell):
                    spells.append(spell)
            start = i
    return spells


def _speeds_look_real(spell):
    """Throw away spells where tracking glitched and someone 'ran' faster than 12 m/s."""
    moves = np.diff(spell.defenders, axis=0)[:, spell.defender_on]
    moves_att = np.diff(spell.attackers, axis=0)[:, spell.attacker_on]
    fastest = max(np.abs(moves).max(), np.abs(moves_att).max()) / STEP
    return fastest < 12.0


def shots(match):
    """Every shot in the match: who took it, when, and if it was a goal."""
    found = []
    for e in match.events:
        if e["type"] == "SHOT":
            frame = int(np.searchsorted(match.raw_frame, e["start"]))
            found.append({"team": e["team"], "frame": min(frame, len(match.raw_frame) - 1),
                          "goal": "GOAL" in e["subtype"], "period": e["period"], "raw_frame": e["start"]})
    return found
