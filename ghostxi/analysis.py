"""
Looks at one shot and asks: were the defenders where typical defenders
would have been?

The ghosts set off from where the real defenders stood 8 seconds before
the shot. At the moment of the shot we compare:

  off_by:  how far each real defender is from his own solo ghost (meters)
  danger:  how much of the space in front of goal the attackers would
           win, with the real defense and with the ghost defense
  space:   how far the nearest outfield defender is from the ball

TV footage does not show everyone. A defender can only be compared if
he was on screen at the start (so his ghost has a starting spot) and is
on screen at the shot (so we know where he really was).
"""

import numpy as np

from .danger import danger, space
from .ghost import make_ghosts, ghost_speed

LOOK_BACK = 8.0   # seconds before the shot where the ghosts set off


def start_frame(clip):
    """The frame where the ghosts set off: 8 seconds before the shot, or the start of the clip."""
    return max(0, clip.shot_frame - int(LOOK_BACK * clip.fps))


def ghost_defense(clip, ghosts, frame, speeds):
    """
    The defense at one frame with the ghosts swapped in.

    A defender stands on his ghost if he has one and is on screen now.
    A defender with no ghost stays where he really is.
    A defender who is off screen now is in neither defense (NaN).

    Returns (positions, speeds, swapped). swapped says who was replaced.
    """
    real = clip.defenders[frame]
    swapped = ~np.isnan(real).any(axis=1) & ~np.isnan(ghosts[frame]).any(axis=1)
    spots = np.where(swapped[:, None], ghosts[frame], real)
    moving = np.where(swapped[:, None], ghost_speed(ghosts, frame, clip.fps), speeds["def"][frame])
    return spots, moving, swapped


def analyse_shot(model, clip):
    """Run the ghosts on one clip and measure everything at the moment of the shot."""
    start, shot = start_frame(clip), clip.shot_frame
    speeds = clip.speeds()
    half_second = max(1, round(0.5 * clip.fps))

    # Solo ghosts tell us how far each defender is from where he "should" be.
    solo = make_ghosts(model, clip, start, solo=True, only_frames=[shot])
    off_by = np.linalg.norm(clip.defenders[shot] - solo[shot], axis=-1)   # NaN where we cannot compare

    # The team of ghosts tells us if a typical defense would have been safer.
    # We also need them half a second earlier, to know how fast they are moving.
    team = make_ghosts(model, clip, start, solo=False, only_frames=[shot - half_second, shot])
    spots, moving, swapped = ghost_defense(clip, team, shot, speeds)

    attackers, attacker_speeds = clip.attackers[shot], speeds["att"][shot]
    return {
        "seconds": (shot - start) / clip.fps,
        "compared": int(np.isfinite(off_by).sum()),
        "off_by": [float(v) for v in off_by],
        "danger_real": danger(attackers, attacker_speeds, clip.defenders[shot], speeds["def"][shot]),
        "danger_ghost": danger(attackers, attacker_speeds, spots, moving),
        "space_real": space(clip.ball[shot], clip.defenders[shot], clip.def_keeper),
        "space_ghost": space(clip.ball[shot], spots, clip.def_keeper),
    }
