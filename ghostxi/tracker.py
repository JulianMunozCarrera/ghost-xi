"""
Follows players from frame to frame.

A detector tells us "there is a player at this spot" but not WHO it is.
The tracker links the spots over time: the spot nearest to where a
player was heading is probably that same player.

It also smooths. Spots from a camera wobble by half a meter or more, so
the tracker does not jump straight to each new spot. It moves part of
the way there. That gives steadier positions and usable speeds.

When a player leaves the picture for more than a second and comes back,
he gets a new number. That is a real limit of TV footage.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

TRUST_POSITION = 0.5   # how far toward a new spot we move (0 = ignore it, 1 = jump to it)
TRUST_SPEED = 0.15     # how much a surprise changes our idea of the player's speed


class Tracker:
    def __init__(self, fps, max_jump=3.5, forget_after=1.0):
        """
        max_jump:     a spot further than this (meters) from the expected place is someone else
        forget_after: seconds a player may be unseen before we stop waiting for him
        """
        self.fps = fps
        self.max_jump = max_jump
        self.forget_after = forget_after
        self.tracks = {}   # number -> dict(position, velocity, unseen)
        self.next_number = 0

    def _expected(self, track):
        """Where we expect a known player to be in this frame."""
        return track["position"] + track["velocity"] / self.fps * (track["unseen"] + 1)

    def update(self, spots):
        """
        spots: (count, 2) positions on the pitch for this frame, in any order.
        Returns a list with the track number given to each spot.
        After the call, self.tracks[number] holds the smoothed position and speed.
        """
        spots = np.asarray(spots, float).reshape(-1, 2)
        numbers = list(self.tracks)
        given = [-1] * len(spots)

        if numbers and len(spots):
            expected = np.array([self._expected(self.tracks[n]) for n in numbers])
            cost = np.linalg.norm(expected[:, None] - spots[None], axis=-1)
            # Best overall pairing of known players to spots (the Hungarian method).
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= self.max_jump:
                    track = self.tracks[numbers[r]]
                    frames_passed = track["unseen"] + 1
                    surprise = spots[c] - expected[r]
                    track["position"] = expected[r] + TRUST_POSITION * surprise
                    track["velocity"] = track["velocity"] + TRUST_SPEED * surprise * self.fps / frames_passed
                    track["unseen"] = -1   # becomes 0 below
                    given[c] = numbers[r]

        for n in numbers:
            self.tracks[n]["unseen"] += 1
            if self.tracks[n]["unseen"] > self.forget_after * self.fps:
                del self.tracks[n]

        for c in range(len(spots)):
            if given[c] == -1:
                self.tracks[self.next_number] = {"position": spots[c].copy(), "velocity": np.zeros(2), "unseen": 0}
                given[c] = self.next_number
                self.next_number += 1
        return given
