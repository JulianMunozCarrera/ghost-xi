"""
Looks at a shot or goal and asks: were the defenders where a typical
defender would have been?

For the 5 seconds before the shot we let ghost defenders play in place
of the real ones. Then we compare:

  - gap:    how far each real defender was from their ghost (metres)
  - danger: how much of the area near goal the attackers controlled,
            with the real defence and with the ghost defence
"""

import numpy as np

from .control import attack_share, danger_zone
from .data import FPS, STEP, view_for_defence
from .model import ghost_paths

BEFORE = 5.0   # seconds of play before the shot
AFTER = 1.0    # seconds after the shot, so you can see where the ball goes


def clip_around_shot(match, shot):
    """
    Cut out the play around a shot, seen from the defending team's side.
    If the ball was out of play less than 5 s before (a corner, say),
    the clip starts when play restarted. Returns (spell, index of the
    shot frame inside the spell), or (None, None) if there is under 2 s.
    """
    defending = "away" if shot["team"] == "home" else "home"
    last = min(shot["frame"] + int(AFTER * FPS), len(match.ball) - 1)
    while last > shot["frame"] and np.isnan(match.ball[last, 0]):
        last -= 1
    first = shot["frame"] - int(BEFORE * FPS) - 1   # one extra frame to measure starting speeds
    while first <= shot["frame"] - 2 * FPS:
        frames = np.arange(first, last + 1)
        spell = view_for_defence(match, defending, frames)
        if spell is not None and not np.isnan(spell.ball).any():
            return spell, shot["frame"] - first
        first += 1
    return None, None


def as_batch(spell):
    """Wrap one spell as a batch of one clip, the shape the model expects."""
    return {
        "defenders": spell.defenders[None], "attackers": spell.attackers[None], "ball": spell.ball[None],
        "defender_on": spell.defender_on[None], "attacker_on": spell.attacker_on[None], "keeper": spell.keeper[None],
    }


def study(spell, shot_index, net):
    """
    Run the ghosts on one clip and measure everything.

    Returns a dict. Paths start at frame 1 of the spell (frame 0 is only
    there to measure speeds), so 'shot_step' is shot_index - 1.
    """
    batch = as_batch(spell)
    team_ghosts = ghost_paths(net, batch)[0]             # all eleven ghosts playing together
    solo_ghosts = ghost_paths(net, batch, solo=True)[0]  # each ghost next to the real team mates
    real = spell.defenders[1:]
    attackers = spell.attackers[1:]
    on, att_on = spell.defender_on, spell.attacker_on
    shot_step = shot_index - 1

    def speeds(path, before):
        return np.diff(np.concatenate([before[None], path]), axis=0) / STEP

    real_speed = speeds(real, spell.defenders[0])
    attacker_speed = speeds(attackers, spell.attackers[0])
    ghost_speed = speeds(team_ghosts, spell.defenders[0])

    spots, weights = danger_zone()
    danger_real, danger_ghost = [], []
    for t in range(len(real)):
        danger_real.append(attack_share(real[t][on], real_speed[t][on], attackers[t][att_on], attacker_speed[t][att_on], spots, weights))
        danger_ghost.append(attack_share(team_ghosts[t][on], ghost_speed[t][on], attackers[t][att_on], attacker_speed[t][att_on], spots, weights))

    gap = np.linalg.norm(solo_ghosts - real, axis=-1)   # (steps, 11) metres between each defender and their ghost
    return {
        "real": real, "team_ghosts": team_ghosts, "solo_ghosts": solo_ghosts,
        "attackers": attackers, "ball": spell.ball[1:],
        "defender_on": on, "attacker_on": att_on, "keeper": spell.keeper,
        "shot_step": shot_step,
        "gap": gap,
        "gap_at_shot": gap[shot_step],
        "danger_real": np.array(danger_real), "danger_ghost": np.array(danger_ghost),
    }
