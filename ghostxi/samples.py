"""
Cuts matches into training examples for the model.

One example = two moments of the same piece of play:
    "then" (a starting point) and "now" (up to 10 seconds later).
We hide where some defenders are "now" and the model has to guess.
Since we know where they really went, we can score the guess.

This does the same job as tokens.build_tokens, but for hundreds of
moments at once, which makes training fast. A test checks that both
give identical numbers.
"""

import numpy as np

from .pitch import MAX_PLAYERS, speed_from_positions
from .tokens import (TOKENS, FEATURES, FIRST_ATTACKER, FIRST_DEFENDER, POSITION_SCALE,
                     SPEED_SCALE, MOVE_SCALE, NEAR_SCALE, TIME_SCALE, MAX_SECONDS)

PER_SECOND = 5   # how many "now" moments we take from each second of play


def speed_reach(fps):
    """How many frames each side we look when working out a speed. About 0.12 seconds."""
    return max(1, round(0.12 * fps))


class MatchSamples:
    def __init__(self, match):
        self.match = match
        self.fps = match.fps
        self.every = max(1, round(match.fps / PER_SECOND))   # use one frame in this many
        every = self.every

        # Positions we trust: only real sightings. Everything else is NaN.
        self.xy = np.where(match.seen[..., None], match.xy, np.nan).astype(np.float32)
        self.speed = speed_from_positions(self.xy, match.fps, speed_reach(match.fps))
        self.ball_speed = speed_from_positions(match.ball, match.fps, speed_reach(match.fps), top_speed=40.0)
        # Does this match show all 22 players (stadium tracking) or only the TV picture?
        self.full_view = bool(np.all(match.seen == ~np.isnan(match.xy[..., 0])))

        frames = len(match.ball)
        alive = ~np.isnan(match.ball[:, 0])

        # For every frame: the frame where the current stretch of live play began.
        # We never pair a "then" and a "now" across a stoppage.
        began = np.zeros(frames, dtype=np.int64)
        for f in range(frames):
            if not alive[f]:
                began[f] = f + 1
            elif f > 0 and alive[f - 1] and match.period[f] == match.period[f - 1]:
                began[f] = began[f - 1]
            else:
                began[f] = f

        usable = alive & (match.possession >= 0)
        times = np.where(usable)[0]
        times = times[times % every == 0]
        # A moment is only useful if we can see some defenders and at least one attacker.
        attacking = match.possession[times].astype(np.int64)
        seen_count = match.seen[times].sum(axis=2)
        rows = np.arange(len(times))
        enough = (seen_count[rows, 1 - attacking] >= 3) & (seen_count[rows, attacking] >= 1)
        longest = np.minimum(times - began[times], int(MAX_SECONDS * self.fps))
        keep = (longest >= every) & enough
        self.times = times[keep]         # frames we can use as "now"
        self.longest = longest[keep]     # how many frames back "then" may be

        # Small lookup tables so we can fetch things for many frames at once.
        self.direction = np.array([[match.direction[(p, t)] for t in (0, 1)] for p in (1, 2)], dtype=np.float32)
        self.keeper = np.array([[match.keeper[(p, t)] for t in (0, 1)] for p in (1, 2)])

    def _team(self, now, then, team, sign):
        """Fetch one team at both moments, turned so the attack goes right."""
        rows = np.arange(len(now))
        xy_now, xy_then = self.xy[now, team], self.xy[then, team]
        v_now, v_then = self.speed[now, team], self.speed[then, team]

        # Squads have more than 11 slots (substitutes). Players seen now go
        # first, then players seen only then, and we keep the first 11.
        rank = np.isnan(xy_now[..., 0]) * 2 + np.isnan(xy_then[..., 0])
        order = np.argsort(rank, axis=1, kind="stable")[:, :MAX_PLAYERS]
        pick = lambda a: a[rows[:, None], order] * sign[:, None, None]
        keeper = self.keeper[self.match.period[now] - 1, team]
        return pick(xy_now), pick(v_now), pick(xy_then), pick(v_then), order == keeper[:, None]

    def build(self, now, back, hidden_fn, rng=None, tv_chance=0.0, mirror=False):
        """
        now:  frame numbers for the "now" moments
        back: how many frames earlier each "then" moment is
        hidden_fn: gets a (count, 11) True/False table of defenders we could
                   hide (seen at both moments) and returns which ones to hide
        tv_chance: how often we pretend a full-view match was filmed by a TV
                   camera, by removing players far from the ball
        mirror: flip some examples top to bottom, which doubles our data for free

        Returns tokens, present, hidden, target (all with one row per moment).
        """
        now = np.asarray(now)
        then = now - np.asarray(back)
        count = len(now)
        seconds = np.asarray(back) / self.fps

        attacking = self.match.possession[now].astype(np.int64)
        sign = self.direction[self.match.period[now] - 1, attacking]

        tokens = np.zeros((count, TOKENS, FEATURES), dtype=np.float32)
        present = np.zeros((count, TOKENS), dtype=bool)
        hidden = np.zeros((count, TOKENS), dtype=bool)
        target = np.zeros((count, TOKENS, 2), dtype=np.float32)

        # Ball
        ball_now = self.match.ball[now] * sign[:, None]
        ball_then = self.match.ball[then] * sign[:, None]
        tokens[:, 0, 0] = 1
        tokens[:, 0, 5:7] = ball_now / POSITION_SCALE
        tokens[:, 0, 7:9] = self.ball_speed[now] * sign[:, None] / SPEED_SCALE
        tokens[:, 0, 9:11] = ball_then / POSITION_SCALE
        tokens[:, 0, 11:13] = self.ball_speed[then] * sign[:, None] / SPEED_SCALE
        present[:, 0] = True

        if tv_chance > 0:
            # A pretend TV camera: it shows a stretch of the pitch around the
            # ball, between 36 and 76 meters long. It is aimed a little off the
            # ball, and it is aimed again for the "then" moment.
            pretend = rng.random(count) < tv_chance
            half = rng.uniform(18.0, 38.0, count)
            aim_now = ball_now[:, 0] + rng.uniform(-8.0, 8.0, count)
            aim_then = ball_then[:, 0] + rng.uniform(-8.0, 8.0, count)

        for kind, team, first in ((1, attacking, FIRST_ATTACKER), (2, 1 - attacking, FIRST_DEFENDER)):
            xy_now, v_now, xy_then, v_then, is_keeper = self._team(now, then, team, sign)
            block = slice(first, first + MAX_PLAYERS)
            seen_now = ~np.isnan(xy_now[..., 0])
            seen_then = ~np.isnan(xy_then[..., 0])
            if tv_chance > 0:
                seen_now &= ~pretend[:, None] | (np.abs(xy_now[..., 0] - aim_now[:, None]) <= half[:, None])
                seen_then &= ~pretend[:, None] | (np.abs(xy_then[..., 0] - aim_then[:, None]) <= half[:, None])
            xy_now, xy_then = np.nan_to_num(xy_now), np.nan_to_num(xy_then)

            hide = np.zeros_like(seen_now)
            if kind == 2:
                # We can only hide (and score) a defender we saw at both moments.
                hide = hidden_fn(seen_now & seen_then) & seen_now & seen_then
                hidden[:, block] = hide
                target[:, block] = np.where(hide[..., None], (xy_now - xy_then) / MOVE_SCALE, 0)
            show_now = (seen_now & ~hide)[..., None]     # his "now" goes into the token
            show_then = seen_then[..., None]             # his "then" goes into the token

            tokens[:, block, kind] = 1
            tokens[:, block, 3] = is_keeper
            tokens[:, block, 4] = hide
            tokens[:, block, 5:7] = np.where(show_now, xy_now / POSITION_SCALE, 0)
            tokens[:, block, 7:9] = np.where(show_now, v_now / SPEED_SCALE, 0)
            tokens[:, block, 9:11] = np.where(show_then, xy_then / POSITION_SCALE, 0)
            tokens[:, block, 11:13] = np.where(show_then, v_then / SPEED_SCALE, 0)
            tokens[:, block, 16:18] = np.where(show_now, (xy_now - ball_now[:, None]) / NEAR_SCALE, 0)
            tokens[:, block, 18:20] = np.where(show_then, (xy_then - ball_then[:, None]) / NEAR_SCALE, 0)
            tokens[:, block, 20] = ~seen_then
            # A hidden defender needs his "then". Everyone else needs his "now".
            present[:, block] = np.where(hide, seen_then, seen_now)

        tokens[:, :, 13] = (seconds / TIME_SCALE)[:, None]
        tokens[:, :, 14:16] = ((ball_now - ball_then) / NEAR_SCALE)[:, None, :]

        if mirror:
            flip = np.where(rng.random(count) < 0.5, -1.0, 1.0).astype(np.float32)
            tokens[:, :, [6, 8, 10, 12, 15, 17, 19]] *= flip[:, None, None]
            target[:, :, 1] *= flip[:, None]

        tokens[~present] = 0
        return tokens, present, hidden, target

    def random_batch(self, size, rng):
        """A training batch: random moments, random gaps, random defenders hidden."""
        pick = rng.integers(0, len(self.times), size)
        now = self.times[pick]
        steps = self.longest[pick] // self.every
        back = (1 + (rng.random(size) * steps).astype(np.int64)) * self.every

        def choose_hidden(usable):
            hide = np.zeros_like(usable)
            style = rng.random(size)
            noise = rng.random(usable.shape)
            for i in range(size):
                options = np.where(usable[i])[0]
                if len(options) == 0:
                    continue
                if style[i] < 0.35:      # hide one defender
                    hide[i, rng.choice(options)] = True
                elif style[i] < 0.70:    # hide the whole defense
                    hide[i, options] = True
                else:                    # hide a random handful
                    chosen = options[noise[i, options] < rng.uniform(0.2, 0.8)]
                    hide[i, chosen if len(chosen) else rng.choice(options)] = True
            return hide

        # Stadium tracking shows everyone. Half the time we pretend it was
        # a TV camera, so the model learns to cope with missing players.
        return self.build(now, back, choose_hidden, rng, tv_chance=0.5 if self.full_view else 0.0, mirror=True)
