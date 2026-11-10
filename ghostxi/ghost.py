"""
Makes the ghosts.

Given a clip and a starting frame, this asks the model, for every later
frame: "these defenders stood HERE at the start. The ball and attackers
have since moved like THIS. Where would typical pro defenders be now?"

Two kinds of ghost:
  team ghost: all defenders are hidden at once. The model only sees the
              ball and the attackers. This is the "Ghost XI".
  solo ghost: one defender is hidden at a time. The model also sees his
              real teammates. This answers "given what everyone else
              did, where should THIS player have been?"

A defender only gets a ghost if he was on screen at the starting frame.
Without that we do not know where his ghost should set off from.
"""

import numpy as np

from .tokens import build_tokens, moves_to_positions, MAX_SECONDS


def make_ghosts(model, clip, start, solo=False, only_frames=None):
    """
    Returns ghost positions with shape (frames, defenders, 2).
    Before and at the start frame the ghosts stand where the real defenders stood.
    A defender who was not on screen at the start gets NaN (no ghost).
    Frames more than 10 seconds after the start are NaN (the model is not trained that far).

    only_frames: if you only need a few frames, list them here to save time.
                 All other later frames are left as NaN.
    """
    speeds = clip.speeds()
    frames, count = clip.defenders.shape[:2]
    ghosts = np.full((frames, count, 2), np.nan, dtype=np.float32)
    ghosts[:start + 1] = clip.defenders[:start + 1]
    then = clip.moment(start, speeds)
    can_ghost = ~np.isnan(then["def"]).any(axis=1)   # on screen at the start
    if np.isnan(then["ball"]).any() or not can_ghost.any():
        return ghosts

    last = min(frames - 1, start + int(MAX_SECONDS * clip.fps))
    later = [f for f in range(start + 1, last + 1) if not np.isnan(clip.ball[f]).any()]
    if only_frames is not None:
        later = [f for f in later if f in set(only_frames)]
    if not later:
        return ghosts

    # Each "job" is one question for the model: (frame, which defenders to hide).
    if solo:
        jobs = [(f, [i == d for i in range(count)]) for f in later for d in range(count) if can_ghost[d]]
    else:
        jobs = [(f, list(can_ghost)) for f in later]

    tokens, present = zip(*[build_tokens(clip.moment(f, speeds), then, (f - start) / clip.fps, hide)
                            for f, hide in jobs])
    answers = model.run(np.stack(tokens), np.stack(present))

    for (f, hide), answer in zip(jobs, answers):
        spots = moves_to_positions(answer, then["def"])
        ghosts[f, hide] = spots[hide]
    return ghosts


def ghost_speed(ghosts, frame, fps):
    """
    How fast the ghosts are moving at one frame, in meters per second.

    We compare with where they were half a second earlier. A shorter gap
    would make the speed jumpy, because each frame is a separate guess.
    """
    step = max(1, round(0.5 * fps))
    earlier = max(frame - step, 0)
    if earlier == frame:
        return np.zeros_like(ghosts[frame])
    speed = (ghosts[frame] - ghosts[earlier]) / ((frame - earlier) / fps)
    speed = np.nan_to_num(speed)   # no ghost half a second ago: treat him as standing still
    size = np.linalg.norm(speed, axis=-1, keepdims=True)
    return np.where(size > 12.0, speed * 12.0 / np.maximum(size, 1e-9), speed)
