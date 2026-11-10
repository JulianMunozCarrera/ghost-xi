"""
The ghost model.

Idea: watch thousands of real defending situations and learn what a
typical defender does next. Then, in any new situation, we can ask
"where would a typical defender go from here?" That imaginary defender
is the ghost.

How one step works. For every defender, the network looks at:
  - the defender (where they are, how fast they are moving),
  - the ball,
  - every attacker,
  - every team mate,
and answers one question: how should this defender speed up or slow
down over the next 0.2 seconds?

Do that again and again and you get a path. That is a "rollout".
"""

import numpy as np
import torch
import torch.nn as nn

from .data import STEP

POSITION_SCALE = 50.0   # divide metres by this so the numbers are near 1
SPEED_SCALE = 8.0       # same for metres per second
STRONGEST_PUSH = 7.0    # a player can't change speed by more than 7 m/s every second


def small_net(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, outputs), nn.ReLU())


class GhostNet(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.read_attacker = small_net(10, hidden, hidden)
        self.read_mate = small_net(8, hidden, hidden)
        self.decide = nn.Sequential(
            nn.Linear(12 + 4 * hidden, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, me, my_speed, keeper, mates, mate_speeds, mate_on,
                attackers, attacker_speeds, attacker_on, ball, ball_speed):
        """
        Shapes (B = situations in the batch):
          me, my_speed            (B, 11, 2)  the defenders we are moving
          mates, mate_speeds      (B, 11, 2)  what each defender sees of the team
          attackers, ...          (B, 11, 2)
          ball, ball_speed        (B, 2)
          keeper, mate_on, attacker_on  (B, 11) true/false
        Returns the push (acceleration) for each defender, (B, 11, 2).
        """
        P, S = POSITION_SCALE, SPEED_SCALE
        ball_wide = ball[:, None, :]
        to_ball = (ball_wide - me) / P

        about_me = torch.cat([
            me / P, my_speed / S, keeper[..., None].float(),
            to_ball, to_ball.norm(dim=-1, keepdim=True),
            ball_wide.expand_as(me) / P, ball_speed[:, None, :].expand_as(me) / S,
        ], dim=-1)

        # Every defender looks at every attacker. gap[b, i, j] = attacker j as seen from defender i.
        gap = (attackers[:, None, :, :] - me[:, :, None, :]) / P
        attacker_to_ball = (attackers - ball_wide) / P
        about_attackers = torch.cat([
            gap, gap.norm(dim=-1, keepdim=True),
            _for_each_defender(attacker_speeds / S),
            _for_each_defender(attacker_to_ball),
            _for_each_defender(attacker_to_ball.norm(dim=-1, keepdim=True)),
            _for_each_defender(attackers / P),
        ], dim=-1)
        attack_summary = _summarise(self.read_attacker(about_attackers), attacker_on[:, None, :].expand(-1, 11, -1))

        # Same for team mates, but a defender does not count as their own mate.
        gap = (mates[:, None, :, :] - me[:, :, None, :]) / P
        about_mates = torch.cat([
            gap, gap.norm(dim=-1, keepdim=True),
            _for_each_defender(mate_speeds / S),
            _for_each_defender(keeper[..., None].float()),
            _for_each_defender(mates / P),
        ], dim=-1)
        not_me = ~torch.eye(11, dtype=torch.bool, device=me.device)[None]
        mate_summary = _summarise(self.read_mate(about_mates), mate_on[:, None, :] & not_me)

        push = self.decide(torch.cat([about_me, attack_summary, mate_summary], dim=-1))
        return STRONGEST_PUSH * torch.tanh(push)


def _for_each_defender(x):
    """Copy (B, 11, k) into (B, 11, 11, k) so every defender gets the full list."""
    return x[:, None, :, :].expand(-1, 11, -1, -1)


def _summarise(x, keep):
    """
    Squash a list of players into one summary, ignoring empty slots.
    We keep both the average and the biggest value of each number.
    This way the order of the players does not matter.
    """
    keep = keep[..., None]
    count = keep.sum(dim=2).clamp(min=1)
    average = (x * keep).sum(dim=2) / count
    biggest = x.masked_fill(~keep, -1e9).max(dim=2).values.clamp(min=0)
    return torch.cat([average, biggest], dim=-1)


def speeds(positions):
    """Speed at every frame, from how far things moved since the frame before. Works on (.., frames, .., 2)."""
    return (positions[:, 1:] - positions[:, :-1]) / STEP


def rollout(net, defenders, attackers, ball, defender_on, attacker_on, keeper, solo=False):
    """
    Let the ghosts play.

    All inputs are real tracking data with shape (B, frames, ...).
    Frame 0 is only used to measure starting speeds. The ghosts start
    at frame 1 in the real positions, and from then on they move on
    their own. The attackers and the ball always follow the real data.

    solo=False: all eleven ghosts play together and see each other.
    solo=True:  each ghost plays next to the REAL team mates. This shows
                what one defender should have done given what the
                others really did.

    Returns ghost positions for frames 1 onwards, (B, frames - 1, 11, 2).
    """
    attacker_speeds, ball_speeds, real_speeds = speeds(attackers), speeds(ball), speeds(defenders)
    position = defenders[:, 1]
    speed = real_speeds[:, 0]
    path = [position]
    for t in range(1, defenders.shape[1] - 1):
        mates, mate_speeds = (defenders[:, t], real_speeds[:, t - 1]) if solo else (position, speed)
        push = net(position, speed, keeper, mates, mate_speeds, defender_on,
                   attackers[:, t], attacker_speeds[:, t - 1], attacker_on, ball[:, t], ball_speeds[:, t - 1])
        speed = speed + STEP * push
        position = position + STEP * speed
        path.append(position)
    return torch.stack(path, dim=1)


def ghost_paths(net, spell_arrays, solo=False):
    """Handy wrapper for numpy arrays. Takes a dict of arrays, returns a numpy array."""
    tensors = {k: torch.as_tensor(v) for k, v in spell_arrays.items()}
    net.eval()
    with torch.no_grad():
        return rollout(net, tensors["defenders"], tensors["attackers"], tensors["ball"], tensors["defender_on"],
                       tensors["attacker_on"], tensors["keeper"], solo=solo).numpy()


def load(path):
    net = GhostNet()
    net.load_state_dict(torch.load(path))
    net.eval()
    return net
