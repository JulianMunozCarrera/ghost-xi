"""
A simple model of a TV camera.

The main camera at a football match sits high in the stand on the
halfway line. It can turn left and right (pan), tilt up and down, and
zoom. It does not move from its spot.

This file does the maths that links a spot on the pitch (in metres)
to a pixel in the picture, and back again.
"""

from dataclasses import dataclass

import numpy as np

from .pitch import HALF_WIDTH

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# Where the camera stands: on the halfway line, 25 m behind the touchline, 18 m up.
CAMERA_SPOT = np.array([0.0, -HALF_WIDTH - 25.0, 18.0])


@dataclass
class Camera:
    pan: float     # radians. 0 = looking straight across the pitch. Positive = turned right.
    tilt: float    # radians below the horizon.
    zoom: float    # focal length in pixels. Bigger = more zoomed in.

    def axes(self):
        """Three arrows: where the camera looks, which way is right, which way is down in the picture."""
        forward = np.array([np.sin(self.pan) * np.cos(self.tilt), np.cos(self.pan) * np.cos(self.tilt), -np.sin(self.tilt)])
        right = np.array([np.cos(self.pan), -np.sin(self.pan), 0.0])
        down = -np.cross(right, forward)
        return forward, right, down

    def to_pixels(self, points):
        """
        Pitch to picture. points is (n, 2) for spots on the grass, or (n, 3)
        if you also give a height. Returns (n, 2) pixels and a True/False
        list saying which points are in front of the camera.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.shape[1] == 2:
            points = np.concatenate([points, np.zeros((len(points), 1))], axis=1)
        forward, right, down = self.axes()
        from_camera = points - CAMERA_SPOT
        depth = from_camera @ forward
        in_front = depth > 1.0
        depth = np.where(in_front, depth, 1.0)
        u = IMAGE_WIDTH / 2 + self.zoom * (from_camera @ right) / depth
        v = IMAGE_HEIGHT / 2 + self.zoom * (from_camera @ down) / depth
        return np.stack([u, v], axis=1), in_front

    def to_pitch(self, pixels, height=0.0):
        """
        Picture to pitch. Each pixel is a line of sight from the camera.
        We follow that line until it hits the grass.
        height: stop the line this many metres above the grass instead.
        Useful for a player's head, which is about 1.8 m up.
        """
        pixels = np.asarray(pixels, dtype=np.float64)
        forward, right, down = self.axes()
        rays = (forward[None, :]
                + ((pixels[:, 0] - IMAGE_WIDTH / 2) / self.zoom)[:, None] * right[None, :]
                + ((pixels[:, 1] - IMAGE_HEIGHT / 2) / self.zoom)[:, None] * down[None, :])
        falling = np.minimum(rays[:, 2], -1e-6)   # the line must point down to reach the grass
        reach = (height - CAMERA_SPOT[2]) / falling
        return (CAMERA_SPOT[None, :] + reach[:, None] * rays)[:, :2]

    def as_list(self):
        return np.array([self.pan, self.tilt, self.zoom])


def look_at(spot, zoom):
    """The camera settings that put a spot on the pitch in the middle of the picture."""
    away = np.array([spot[0], spot[1], 0.0]) - CAMERA_SPOT
    pan = np.arctan2(away[0], away[1])
    tilt = np.arctan2(-away[2], np.hypot(away[0], away[1]))
    return Camera(pan, tilt, zoom)
