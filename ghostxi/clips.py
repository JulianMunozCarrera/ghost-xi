"""
A Clip is a short piece of play that ends with a shot.

It is the common shape for everything downstream: the ghost maker, the
danger score and the website all work on Clips. It does not matter if
the clip came from a full match or from a goal tracked off TV footage.

Inside a clip the attacking team always attacks to the right.
A position of NaN means the player was not on screen in that frame.
"""

from dataclasses import dataclass, field

import numpy as np

from .pitch import speed_from_positions
from .samples import speed_reach


@dataclass
class Clip:
    name: str
    source: str
    fps: float
    ball: np.ndarray        # (frames, 2)
    attackers: np.ndarray   # (frames, attackers, 2)
    defenders: np.ndarray   # (frames, defenders, 2)
    att_keeper: int         # which attacker is a goalkeeper, or -1 if he is not in the clip
    def_keeper: int         # which defender is the goalkeeper, or -1
    shot_frame: int         # the frame of the shot
    def_numbers: list = field(default_factory=list)   # shirt numbers of the defenders, if we know them
    att_team: str = ""      # team names, if we know them
    def_team: str = ""
    goal: bool = True       # did the shot go in?

    def __post_init__(self):
        # Round every position to a whole centimeter. Tracking is nowhere
        # near that exact anyway, and it lets the website store the very
        # same numbers, so Python and the website always agree.
        for name in ("ball", "attackers", "defenders"):
            setattr(self, name, (np.round(np.asarray(getattr(self, name), dtype=np.float64) * 100) / 100).astype(np.float32))

    def cut(self, first, last):
        """A shorter copy of this clip, from frame 'first' to frame 'last' (both included)."""
        return Clip(self.name, self.source, self.fps, self.ball[first:last + 1], self.attackers[first:last + 1],
                    self.defenders[first:last + 1], self.att_keeper, self.def_keeper, self.shot_frame - first,
                    self.def_numbers, self.att_team, self.def_team, self.goal)

    def speeds(self):
        """How fast everything moves, worked out from the positions."""
        reach = speed_reach(self.fps)
        return {
            "ball": speed_from_positions(self.ball, self.fps, reach, top_speed=40.0),
            "att": speed_from_positions(self.attackers, self.fps, reach),
            "def": speed_from_positions(self.defenders, self.fps, reach),
        }

    def moment(self, frame, speeds):
        """Everything about one frame, in the shape tokens.build_tokens wants."""
        return {
            "ball": self.ball[frame], "ball_v": speeds["ball"][frame],
            "att": self.attackers[frame], "att_v": speeds["att"][frame],
            "def": self.defenders[frame], "def_v": speeds["def"][frame],
            "att_keeper": self.att_keeper, "def_keeper": self.def_keeper,
        }


def clip_from_match(match, start, shot_frame, name, goal=True):
    """
    Cut a clip out of a full match, from frame 'start' to the shot.
    The team with the ball at the shot is the attacking team.
    Only real sightings go into the clip. Guessed positions become NaN.
    """
    attacking = int(match.possession[shot_frame])
    assert attacking in (0, 1), "nobody has the ball at this frame"
    defending = 1 - attacking
    period = int(match.period[shot_frame])
    sign = match.direction[(period, attacking)]
    frames = slice(start, shot_frame + 1)

    def team(index):
        real = np.where(match.seen[frames, index][..., None], match.xy[frames, index], np.nan)
        # Keep the players who are on the pitch at the shot and on screen at least once.
        on_pitch = ~np.isnan(match.xy[shot_frame, index, :, 0])
        keep = np.where(on_pitch & ~np.isnan(real[..., 0]).all(axis=0))[0]
        keeper_slot = match.keeper[(period, index)]
        keeper = int(np.where(keep == keeper_slot)[0][0]) if keeper_slot in keep else -1
        return real[:, keep] * sign, keeper, [int(n) for n in match.numbers[index, keep]]

    attackers, att_keeper, _ = team(attacking)
    defenders, def_keeper, def_numbers = team(defending)
    return Clip(name, match.name, float(match.fps), match.ball[frames] * sign, attackers, defenders,
                att_keeper, def_keeper, shot_frame - start, def_numbers,
                match.teams[attacking], match.teams[defending], goal)
