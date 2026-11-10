"""
Loads the three sample matches that Metrica Sports shared publicly.
https://github.com/metrica-sports/sample-data

Each match gives us, 25 times per second, where all 22 players and the
ball are. It also has a list of events (passes, shots, goals).

The raw files use two different formats, so there are two readers here.
Both end up as the same Match object.
"""

import json
import os
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .pitch import LENGTH, WIDTH, HALF_LENGTH

FPS = 25

# Events that tell us a team has the ball.
HAS_BALL = {"PASS", "SHOT", "CARRY", "RECOVERY", "SET PIECE"}


@dataclass
class Match:
    """
    One full match in a common shape. Every data source ends up as this.

    "seen" matters for matches tracked from TV. A TV camera only shows
    part of the pitch, so the tracking company fills in the players who
    are off screen with a guess. seen says which positions are real.
    We only ever learn from, and judge, positions that are real.
    """
    name: str
    fps: int
    xy: np.ndarray         # (frames, 2 teams, players, 2) in meters. NaN = not on the pitch.
    ball: np.ndarray       # (frames, 2) in meters. NaN = ball is out of play.
    period: np.ndarray     # (frames,) 1 or 2
    possession: np.ndarray  # (frames,) 0 = home has the ball, 1 = away, -1 = nobody / out of play
    direction: dict        # direction[(period, team)] is +1 if that team attacks right, else -1
    keeper: dict           # keeper[(period, team)] is the player slot of the goalkeeper
    shots: list = field(default_factory=list)  # dicts with frame, team, goal, penalty
    seen: np.ndarray = None     # (frames, 2, players) True where the position is a real sighting
    numbers: np.ndarray = None  # (2, players) shirt numbers, 0 if unknown
    teams: tuple = ("Home", "Away")
    source: str = "metrica"     # which dataset this match came from
    label: str = ""             # short text for the website, like "Brisbane FC 2-3 Adelaide United"

    def __post_init__(self):
        if self.seen is None:
            # Stadium tracking sees everyone who is on the pitch.
            self.seen = ~np.isnan(self.xy[..., 0])
        if self.numbers is None:
            self.numbers = np.zeros(self.xy.shape[1:3], dtype=np.int64)
        if not self.label:
            self.label = self.name


def _to_meters(x, y):
    """Metrica stores 0..1 with (0, 0) at the top left. We want meters from the middle."""
    return (x - 0.5) * LENGTH, (0.5 - y) * WIDTH


def _finish(name, raw_xy, raw_ball, period, events):
    """Shared last steps: convert units, then work out possession, direction, keepers and shots."""
    xy = np.stack(_to_meters(raw_xy[..., 0], raw_xy[..., 1]), axis=-1).astype(np.float32)
    ball = np.stack(_to_meters(raw_ball[:, 0], raw_ball[:, 1]), axis=-1).astype(np.float32)
    frames = len(ball)
    alive = ~np.isnan(ball[:, 0])

    # Who has the ball? Walk through the events in order. The team of the
    # latest "has the ball" event keeps possession until the next one.
    possession = np.full(frames, -1, dtype=np.int8)
    events = sorted(events, key=lambda e: e["start"])
    owner_changes = [(e["start"], e["team"]) for e in events if e["type"] in HAS_BALL]
    for i, (start, team) in enumerate(owner_changes):
        end = owner_changes[i + 1][0] if i + 1 < len(owner_changes) else frames
        possession[start:end] = team
    possession[~alive] = -1

    direction, keeper = {}, {}
    for p in (1, 2):
        in_period = np.where((period == p) & alive)[0]
        start = in_period[:FPS * 2]  # the first two seconds of the half
        for team in (0, 1):
            # At kick off every team stands in its own half.
            # A team standing on the left attacks to the right.
            mean_x = np.nanmean(xy[start, team, :, 0])
            direction[(p, team)] = 1 if mean_x < 0 else -1

            # The goalkeeper is the player who stays closest to his own goal.
            own_goal_x = -HALF_LENGTH * direction[(p, team)]
            sample = in_period[::FPS]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # empty slots (unused subs) are all NaN, that is fine
                distance = np.nanmean(np.hypot(xy[sample, team, :, 0] - own_goal_x, xy[sample, team, :, 1]), axis=0)
            keeper[(p, team)] = int(np.nanargmin(distance))

    shots = []
    penalty_frames = [e["start"] for e in events if e["type"] == "SET PIECE" and "PENALTY" in e["subtype"]]
    for e in events:
        if e["type"] != "SHOT":
            continue
        shots.append({
            "frame": e["start"],
            "team": e["team"],
            "goal": "GOAL" in e["subtype"],
            "penalty": any(abs(e["start"] - f) <= FPS for f in penalty_frames),
        })

    return Match(name, FPS, xy, ball, period.astype(np.int8), possession, direction, keeper, shots)


def load_csv_match(folder, number):
    """Reader for Sample_Game_1 and Sample_Game_2 (CSV files)."""
    def read_team(side):
        path = os.path.join(folder, f"Sample_Game_{number}_RawTrackingData_{side}_Team.csv")
        table = pd.read_csv(path, skiprows=2)
        columns = list(table.columns)
        players = [c for c in columns if c.startswith("Player")]
        # Each player name sits above his x column. The y column is right after it.
        xy = np.stack([table.iloc[:, [columns.index(p), columns.index(p) + 1]].to_numpy(dtype=np.float64)
                       for p in players], axis=1)
        ball_at = columns.index("Ball")
        ball = table.iloc[:, [ball_at, ball_at + 1]].to_numpy(dtype=np.float64)
        return xy, ball, table["Period"].to_numpy(), players

    home, ball, period, _ = read_team("Home")
    away, _, _, _ = read_team("Away")

    # Both teams go in one array. Pad the smaller squad with NaN.
    slots = max(home.shape[1], away.shape[1])
    raw = np.full((len(ball), 2, slots, 2), np.nan)
    raw[:, 0, :home.shape[1]] = home
    raw[:, 1, :away.shape[1]] = away

    table = pd.read_csv(os.path.join(folder, f"Sample_Game_{number}_RawEventsData.csv"))
    events = [{
        "team": 0 if row["Team"] == "Home" else 1,
        "type": row["Type"],
        "subtype": row["Subtype"] if isinstance(row["Subtype"], str) else "",
        "start": int(row["Start Frame"]) - 1,  # the file counts from 1, we count from 0
    } for _, row in table.iterrows()]

    return _finish(f"Metrica sample match {number}", raw, ball, period, events)


def load_epts_match(folder, number=3):
    """
    Reader for Sample_Game_3 (FIFA EPTS format).

    The tracking file has one line per frame: "frame:x,y;x,y;...:ballx,bally".
    Which player each "x,y" belongs to is written in the metadata file,
    and it changes during the match when there is a substitution.
    """
    root = ET.parse(os.path.join(folder, f"Sample_Game_{number}_metadata.xml")).getroot()

    team_ids = [t.get("id") for t in root.iter("Team")]
    player_team = {p.get("id"): team_ids.index(p.get("teamId")) for p in root.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}

    # Give every player a fixed slot inside his team.
    slot = {}
    for player, team in player_team.items():
        slot[player] = sum(1 for other in slot if player_team[other] == team)
    slots = max(slot.values()) + 1

    # For each block of frames, the list of (team, slot) in the order they appear on a line.
    layouts = []
    for spec in root.iter("DataFormatSpecification"):
        order = []
        for ref in spec.iter("PlayerChannelRef"):
            channel = ref.get("playerChannelId")
            if channel.endswith("_x"):
                player = channel_player[channel]
                order.append((player_team[player], slot[player]))
        layouts.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))

    params = {p.find("Name").text: p.find("Value").text for p in root.iter("ProviderParameter")}
    first_half_end = int(params["first_half_end"])

    with open(os.path.join(folder, f"Sample_Game_{number}_tracking.txt")) as f:
        lines = f.read().splitlines()
    raw = np.full((len(lines), 2, slots, 2), np.nan)
    ball = np.full((len(lines), 2), np.nan)
    period = np.ones(len(lines), dtype=np.int8)
    first_frame = int(lines[0].split(":")[0])

    for i, line in enumerate(lines):
        frame_text, players_text, ball_text = line.split(":")
        frame = int(frame_text)
        assert frame == first_frame + i, "frames should be in order with none missing"
        period[i] = 1 if frame <= first_half_end else 2
        order = next(o for start, end, o in layouts if start <= frame <= end)
        values = players_text.split(";")
        assert len(values) == len(order)
        for (team, s), value in zip(order, values):
            x, y = value.split(",")
            raw[i, team, s] = float(x), float(y)
        bx, by = ball_text.split(",")
        ball[i] = float(bx), float(by)

    def subtype_text(sub):
        if sub is None:
            return ""
        if isinstance(sub, dict):
            sub = [sub]
        return "-".join(s["name"] for s in sub)

    with open(os.path.join(folder, f"Sample_Game_{number}_events.json")) as f:
        events = [{
            "team": team_ids.index(e["team"]["id"]),
            "type": e["type"]["name"],
            "subtype": subtype_text(e["subtypes"]),
            "start": e["start"]["frame"] - first_frame,
        } for e in json.load(f)["data"]]

    return _finish(f"Metrica sample match {number}", raw, ball, period, events)


def load_all(data_folder):
    """Load all three matches. data_folder is the 'data' folder of the Metrica repo."""
    return [
        load_csv_match(os.path.join(data_folder, "Sample_Game_1"), 1),
        load_csv_match(os.path.join(data_folder, "Sample_Game_2"), 2),
        load_epts_match(os.path.join(data_folder, "Sample_Game_3"), 3),
    ]
