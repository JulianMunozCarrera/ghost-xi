"""
Draws fake TV pictures from real tracking data.

Why fake? Real match video is owned by the leagues, so we can't ship
it or test on it here. Instead we take real player positions and draw
what the main TV camera would see: grass, white lines, and players in
team colours. The camera pans and zooms to follow the ball.

The big plus: we know exactly where every player really is, so we can
measure how well the vision code finds them.
"""

import cv2
import numpy as np

from .camera import Camera, look_at, IMAGE_WIDTH, IMAGE_HEIGHT
from .pitch import pitch_lines, HALF_LENGTH, HALF_WIDTH

# Kit colours in BGR (the order OpenCV uses).
KITS = {
    "defender": (40, 40, 220),        # red
    "attacker": (230, 120, 30),       # blue
    "defender_keeper": (0, 220, 240), # yellow
    "attacker_keeper": (200, 60, 200),# purple
}
PLAYER_HEIGHT = 1.8
PLAYER_WIDTH = 0.6


def follow_ball(ball_path, zoom=1500.0):
    """
    Camera settings for each frame. The camera aims a little towards the
    ball and moves smoothly, like a real camera operator.
    """
    cameras, aim = [], None
    for ball in ball_path:
        target = np.array([ball[0] * 0.85, ball[1] * 0.5])
        aim = target if aim is None else aim + 0.35 * (target - aim)
        cameras.append(look_at(aim, zoom))
    return cameras


def draw_frame(camera, defenders, attackers, ball, defender_keeper=None, attacker_keeper=None, seed=0):
    """
    Draw one picture. defenders and attackers are (n, 2) arrays in metres.
    defender_keeper / attacker_keeper say which row is the goalkeeper.
    """
    random = np.random.default_rng(seed)
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    image[:] = (45, 70, 45)  # dull green-grey for everything outside the pitch

    # Grass in two shades of green, in stripes like a mown pitch.
    for i, start in enumerate(np.arange(-HALF_LENGTH, HALF_LENGTH, 5.25)):
        corners = [(start, -HALF_WIDTH), (start + 5.25, -HALF_WIDTH), (start + 5.25, HALF_WIDTH), (start, HALF_WIDTH)]
        pixels, in_front = camera.to_pixels(_fine(corners + corners[:1]))
        pixels = pixels[in_front]
        if len(pixels) < 3:
            continue
        shade = (50, 150, 60) if i % 2 == 0 else (45, 135, 52)
        cv2.fillPoly(image, [np.round(pixels).astype(np.int32)], shade)

    # White lines. We draw them in short pieces and skip pieces behind the camera.
    for line in pitch_lines():
        pixels, in_front = camera.to_pixels(_fine(line))
        pixels = np.round(pixels).astype(np.int32)
        for i in range(len(pixels) - 1):
            if in_front[i] and in_front[i + 1]:
                cv2.line(image, tuple(pixels[i]), tuple(pixels[i + 1]), (235, 235, 235), 2, cv2.LINE_AA)

    # Players, drawn far to near so closer players cover the ones behind them.
    people = []
    for i, spot in enumerate(defenders):
        people.append((spot, KITS["defender_keeper"] if i == defender_keeper else KITS["defender"]))
    for i, spot in enumerate(attackers):
        people.append((spot, KITS["attacker_keeper"] if i == attacker_keeper else KITS["attacker"]))
    people.sort(key=lambda p: -p[0][1])
    for spot, colour in people:
        feet, _ = camera.to_pixels([spot])
        head, _ = camera.to_pixels([[spot[0], spot[1], PLAYER_HEIGHT]])
        tall = feet[0, 1] - head[0, 1]
        wide = tall * PLAYER_WIDTH / PLAYER_HEIGHT
        centre = (int(round(feet[0, 0])), int(round(feet[0, 1] - tall / 2)))
        cv2.ellipse(image, centre, (max(1, int(round(wide / 2))), max(2, int(round(tall / 2)))), 0, 0, 360, colour, -1, cv2.LINE_AA)

    if ball is not None and not np.isnan(ball[0]):
        pixel, _ = camera.to_pixels([[ball[0], ball[1], 0.11]])
        cv2.circle(image, (int(round(pixel[0, 0])), int(round(pixel[0, 1]))), 3, (250, 250, 250), -1, cv2.LINE_AA)

    # Real pictures are never perfectly clean, so add a bit of blur and grain.
    image = cv2.GaussianBlur(image, (3, 3), 0)
    grain = random.normal(0, 4, image.shape)
    return np.clip(image.astype(np.float64) + grain, 0, 255).astype(np.uint8)


def _fine(points, gap=1.0):
    """Add extra points along a path, about every metre, so curves and long lines draw correctly."""
    points = np.asarray(points, dtype=np.float64)
    out = [points[:1]]
    for a, b in zip(points[:-1], points[1:]):
        pieces = max(1, int(np.ceil(np.linalg.norm(b - a) / gap)))
        out.append(a + (b - a) * (np.arange(1, pieces + 1) / pieces)[:, None])
    return np.concatenate(out)
