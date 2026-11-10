"""
Turns a moment of a match into the numbers the model reads.

The model looks at a moment as a list of up to 23 "tokens":
    1 ball, up to 11 attackers, up to 11 defenders.

Each token is 21 numbers. They describe two points in time:
    "then" = a moment in the past (the starting point)
    "now"  = the moment we care about, a few seconds later

For a defender we want a ghost for, we HIDE where he is now. The model
only gets where he was "then". Its job is to say where he should be now,
after seeing how the ball and everyone else moved in between.

TV cameras do not show everyone. So a player can be missing:
  - not on screen now          -> he gets no token at all
  - on screen now but not then -> he gets a token with his "then" left
                                  blank, and a flag that says so
A defender we want a ghost for must have been on screen "then".
In all the inputs here, a position of NaN means "not on screen".

This file is the single place that defines the 21 numbers. The website
has the same layout written in JavaScript (web/engine.js), and a test
checks that both give the same answer.
"""

import numpy as np

from .pitch import MAX_PLAYERS

TOKENS = 1 + 2 * MAX_PLAYERS   # 23
FEATURES = 21

# Where each group of tokens sits in the list.
BALL = 0
FIRST_ATTACKER = 1
FIRST_DEFENDER = 1 + MAX_PLAYERS

# We divide by these so all numbers are roughly between -1 and 1.
# Neural networks learn better that way.
POSITION_SCALE = 50.0   # meters
SPEED_SCALE = 8.0       # meters per second
TIME_SCALE = 10.0       # seconds
MOVE_SCALE = 10.0       # meters. The model answers "how far did he move" in these units.
NEAR_SCALE = 20.0       # meters. Used for "how far from the ball" numbers.

MAX_SECONDS = 10.0      # the longest gap between "then" and "now" the model is trained on


def one_token(kind, is_keeper, hidden, now_xy, now_v, then_xy, then_v, seconds, ball_now, ball_then,
              then_missing=False):
    """
    Build the 21 numbers for one ball or player.

    kind:   0 = ball, 1 = attacker, 2 = defender
    hidden: True for a defender we want a ghost for (his "now" is left blank)
    then_missing: True for a player who was not on screen "then" (his "then" is left blank)
    """
    token = np.zeros(FEATURES, dtype=np.float32)
    token[kind] = 1.0                       # numbers 0, 1, 2: what am I?
    token[3] = 1.0 if is_keeper else 0.0    # number 3: am I a goalkeeper?
    token[4] = 1.0 if hidden else 0.0       # number 4: is my "now" hidden?
    if not hidden:
        token[5:7] = np.asarray(now_xy) / POSITION_SCALE   # numbers 5, 6: where I am now
        token[7:9] = np.asarray(now_v) / SPEED_SCALE       # numbers 7, 8: how fast I move now
    if not then_missing:
        token[9:11] = np.asarray(then_xy) / POSITION_SCALE     # numbers 9, 10: where I was then
        token[11:13] = np.asarray(then_v) / SPEED_SCALE        # numbers 11, 12: how fast I moved then
    token[13] = seconds / TIME_SCALE                       # number 13: time between then and now

    # Six numbers about the ball. Defending is mostly about the ball, so
    # we spell these out instead of making the model work them out.
    token[14:16] = (np.asarray(ball_now) - np.asarray(ball_then)) / NEAR_SCALE   # 14, 15: how the ball moved
    if kind != 0:
        if not hidden:
            token[16:18] = (np.asarray(now_xy) - np.asarray(ball_now)) / NEAR_SCALE   # 16, 17: me from the ball, now
        if not then_missing:
            token[18:20] = (np.asarray(then_xy) - np.asarray(ball_then)) / NEAR_SCALE     # 18, 19: me from the ball, then
    token[20] = 1.0 if then_missing else 0.0   # number 20: was I off screen then?
    return token


def build_tokens(now, then, seconds, hidden_defenders):
    """
    Build the full list of tokens for one moment.

    now, then: dicts with these keys (positions in meters, speeds in m/s):
        "ball", "ball_v"            shape (2,)
        "att", "att_v"              shape (attackers, 2)
        "def", "def_v"              shape (defenders, 2)
       plus "att_keeper" and "def_keeper": index of each goalkeeper, or -1.
       A position of NaN means the player was not on screen at that moment.
    hidden_defenders: list of True/False, one per defender.

    Returns (tokens, present). tokens is (23, 21). present is 23 True/False
    values. Empty slots are False and the model ignores them.
    """
    tokens = np.zeros((TOKENS, FEATURES), dtype=np.float32)
    present = np.zeros(TOKENS, dtype=bool)

    balls = (now["ball"], then["ball"])
    tokens[BALL] = one_token(0, False, False, now["ball"], now["ball_v"], then["ball"], then["ball_v"], seconds, *balls)
    present[BALL] = True

    for kind, key, first, hidden in ((1, "att", FIRST_ATTACKER, None), (2, "def", FIRST_DEFENDER, hidden_defenders)):
        count = len(now[key])
        assert count <= MAX_PLAYERS
        for i in range(count):
            hide = bool(hidden[i]) if hidden is not None else False
            seen_then = not np.isnan(then[key][i]).any()
            seen_now = not np.isnan(now[key][i]).any()
            # A hidden defender needs his "then". Everyone else needs his "now".
            if (hide and not seen_then) or (not hide and not seen_now):
                continue
            is_keeper = (i == now.get(key + "_keeper", -1))
            tokens[first + i] = one_token(kind, is_keeper, hide, now[key][i], now[key + "_v"][i],
                                          then[key][i], then[key + "_v"][i], seconds, *balls,
                                          then_missing=not seen_then)
            present[first + i] = True
    return tokens, present


def moves_to_positions(model_output, then_def):
    """The model says how far each hidden defender moved. Add that to where he was."""
    return np.asarray(then_def) + np.asarray(model_output)[FIRST_DEFENDER:FIRST_DEFENDER + len(then_def)] * MOVE_SCALE
