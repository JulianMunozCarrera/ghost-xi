"""
A pretend TV camera, and the math to undo it.

The main camera at a match sits high in the stand at the halfway line.
It turns to follow the ball. A spot on the pitch (in meters) lands on
some pixel in the picture. This file can do that step, and the reverse:
given pixels, work out where on the pitch they are.

The reverse step is the interesting one. If we can find a few known
spots in the picture (a corner of the penalty box, the center spot),
we can work out a 3x3 table of numbers, called a homography, that
turns any pixel into a pitch position.
"""

import numpy as np

from .pitch import HALF_LENGTH, HALF_WIDTH, GOAL_WIDTH

CLOSE_ENOUGH = 6.0    # pixels. A landmark this close to where the table puts it is trusted.
ENOUGH_TRUSTED = 5    # we want at least this many trusted landmarks before we believe a table


class Camera:
    WIDTH = 1280    # picture size in pixels
    HEIGHT = 720
    ZOOM = 2000.0   # bigger = more zoomed in
    SPOT = np.array([0.0, -HALF_WIDTH - 25.0, 18.0])   # 25 m behind the touchline, 18 m up

    def __init__(self):
        self.aim = np.zeros(2)   # the spot on the pitch in the middle of the picture

    def follow(self, ball):
        """Turn a little toward the ball, the way a camera operator does. No sudden jumps."""
        target = np.array([ball[0] * 0.85, ball[1] * 0.4])
        self.aim = self.aim + 0.15 * (target - self.aim)

    def _axes(self):
        """Three arrows: where the camera looks, which way is right, which way is down in the picture."""
        forward = np.array([self.aim[0], self.aim[1], 0.0]) - self.SPOT
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right = right / np.linalg.norm(right)
        down = np.cross(forward, right)
        return forward, right, down

    def project(self, points):
        """
        Pitch to picture. points is (count, 2) in meters.
        Returns the pixels (count, 2) and which points are inside the picture.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        forward, right, down = self._axes()
        from_camera = np.concatenate([points, np.zeros((len(points), 1))], axis=1) - self.SPOT
        depth = from_camera @ forward
        safe = np.where(depth > 1.0, depth, 1.0)
        # Things further away get divided by a bigger number, so they look smaller.
        u = self.WIDTH / 2 + self.ZOOM * (from_camera @ right) / safe
        v = self.HEIGHT / 2 + self.ZOOM * (from_camera @ down) / safe
        inside = (depth > 1.0) & (u >= 0) & (u < self.WIDTH) & (v >= 0) & (v < self.HEIGHT)
        return np.stack([u, v], axis=1), inside


def pitch_landmarks():
    """
    Spots on the pitch that are easy to recognise in a picture: where
    lines meet, the penalty spots, points on the center circle.
    Returns (count, 2) in meters.
    """
    L, W = HALF_LENGTH, HALF_WIDTH
    spots = [(0, -W), (0, W), (0, 0), (0, 9.15), (0, -9.15), (9.15, 0), (-9.15, 0)]
    arc_reach = np.sqrt(9.15 ** 2 - 5.5 ** 2)   # where the "D" meets the penalty box
    for side in (-1, 1):
        spots += [
            (side * L, -W), (side * L, W),                                    # corners
            (side * L, -20.16), (side * L, 20.16),                            # penalty box on the goal line
            (side * (L - 16.5), -20.16), (side * (L - 16.5), 20.16),          # penalty box corners
            (side * L, -9.16), (side * L, 9.16),                              # six yard box on the goal line
            (side * (L - 5.5), -9.16), (side * (L - 5.5), 9.16),              # six yard box corners
            (side * L, -GOAL_WIDTH / 2), (side * L, GOAL_WIDTH / 2),          # goal posts
            (side * (L - 11.0), 0),                                           # penalty spot
            (side * (L - 16.5), -arc_reach), (side * (L - 16.5), arc_reach),  # ends of the "D"
            (side * (L - 11.0 - 9.15), 0),                                    # top of the "D"
        ]
    return np.array(spots, dtype=np.float64)


def _tidy(points):
    """Shift and scale points so they sit around zero. This keeps the math below stable."""
    middle = points.mean(axis=0)
    spread = np.sqrt(((points - middle) ** 2).sum(axis=1).mean() / 2) or 1.0
    table = np.array([[1 / spread, 0, -middle[0] / spread], [0, 1 / spread, -middle[1] / spread], [0, 0, 1]])
    return (points - middle) / spread, table


def find_homography(pitch_points, pixel_points):
    """
    Find the 3x3 table that turns pitch positions into pixels.
    Needs at least 4 matching pairs. This is the classic "DLT" method:
    every pair gives two equations, and we solve them all at once.
    """
    pitch_points = np.asarray(pitch_points, dtype=np.float64)
    pixel_points = np.asarray(pixel_points, dtype=np.float64)
    a, tidy_a = _tidy(pitch_points)
    b, tidy_b = _tidy(pixel_points)
    rows = []
    for (x, y), (u, v) in zip(a, b):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    # The answer is the direction the equations squash the most (last row of the SVD).
    _, _, vt = np.linalg.svd(np.array(rows))
    table = np.linalg.inv(tidy_b) @ vt[-1].reshape(3, 3) @ tidy_a
    if abs(table[2, 2]) < 1e-12 or not np.isfinite(table).all():
        raise np.linalg.LinAlgError("these points do not pin down a table")
    return table / table[2, 2]


def apply_homography(table, points):
    """Run points through a 3x3 table. Works for pitch to pixels, or (with the inverse) pixels to pitch."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    moved = np.concatenate([points, np.ones((len(points), 1))], axis=1) @ table.T
    with np.errstate(divide="ignore", invalid="ignore"):   # a bad table can divide by zero. The caller rejects those.
        return moved[:, :2] / moved[:, 2:3]


def find_homography_robust(pitch_points, pixel_points, rng, tries=200):
    """
    Like find_homography, but copes with some landmarks being plain wrong.

    The trick (called RANSAC): pick 4 landmarks at random and build a
    table from only those. Count how many OTHER landmarks agree with
    it. Repeat many times and keep the table most landmarks agree with.
    Wrong landmarks never agree, so they get left out.

    Returns (table, trusted) where trusted says which landmarks agreed.
    Returns (None, None) if we cannot find a table we believe.
    """
    pitch_points = np.asarray(pitch_points, dtype=np.float64)
    pixel_points = np.asarray(pixel_points, dtype=np.float64)
    count = len(pitch_points)
    if count < 4:
        return None, None

    best = None
    for _ in range(tries):
        four = rng.choice(count, 4, replace=False)
        try:
            table = find_homography(pitch_points[four], pixel_points[four])
            off = np.linalg.norm(apply_homography(table, pitch_points) - pixel_points, axis=1)
        except np.linalg.LinAlgError:
            continue
        agree = off < CLOSE_ENOUGH
        if best is None or agree.sum() > best.sum():
            best = agree
    if best is None or best.sum() < min(ENOUGH_TRUSTED, count):
        return None, None

    # Build the final table from every landmark that agreed, then check once more.
    try:
        table = find_homography(pitch_points[best], pixel_points[best])
    except np.linalg.LinAlgError:
        return None, None
    off = np.linalg.norm(apply_homography(table, pitch_points) - pixel_points, axis=1)
    trusted = off < CLOSE_ENOUGH
    if trusted.sum() < min(ENOUGH_TRUSTED, count) or not np.isfinite(table).all():
        return None, None
    try:
        return find_homography(pitch_points[trusted], pixel_points[trusted]), trusted
    except np.linalg.LinAlgError:
        return None, None
