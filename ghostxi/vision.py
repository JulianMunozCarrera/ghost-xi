"""
Turns a TV picture into player positions on the pitch.

Three steps:
  1. Find the camera. We look for the white lines in the picture and
     try camera settings until a drawing of the pitch lines sits right
     on top of them.
  2. Find the players. Anything on the pitch that is not grass and not
     a white line is probably a player. Shirt colour tells us the team.
  3. Put them on the map. A player's feet touch the grass, so we follow
     the line of sight through their feet until it hits the pitch.
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

from .camera import Camera, IMAGE_WIDTH, IMAGE_HEIGHT
from .pitch import pitch_lines, HALF_LENGTH, HALF_WIDTH
from .render import KITS, PLAYER_HEIGHT, PLAYER_WIDTH, _fine

# Points along every pitch line, about one per metre. Made once and reused.
LINE_POINTS = np.concatenate([_fine(line) for line in pitch_lines()])
FAR_AWAY = 25.0  # pixels. A line point further than this from any white pixel just counts as "missing".


def white_line_mask(image):
    """True for pixels that look like white paint: bright and with almost no colour."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] < 60) & (hsv[..., 2] > 170)


def grass_mask(image):
    """True for pixels that look like grass: green and fairly strong in colour."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (hsv[..., 0] > 35) & (hsv[..., 0] < 85) & (hsv[..., 1] > 80)


def _misfit(settings, distance_map):
    """
    How badly do these camera settings explain the picture?
    We draw the pitch lines with these settings and check how far each
    drawn point is from the nearest real white pixel. Small = good fit.
    """
    camera = Camera(settings[0], settings[1], settings[2] * 1000.0)
    pixels, in_front = camera.to_pixels(LINE_POINTS)
    u, v = np.round(pixels[:, 0]).astype(int), np.round(pixels[:, 1]).astype(int)
    visible = in_front & (u >= 0) & (u < IMAGE_WIDTH) & (v >= 0) & (v < IMAGE_HEIGHT)
    if visible.sum() < 40:
        return FAR_AWAY  # hardly any lines in view, so these settings can't be checked
    return float(distance_map[v[visible], u[visible]].mean())


def find_camera(image, guess=None):
    """
    Work out the camera's pan, tilt and zoom from one picture.

    guess: the camera from the frame before. Cameras move smoothly, so
    starting from the last answer is fast and safe. With no guess we try
    a coarse set of settings first and then fine tune the best one.
    """
    lines = white_line_mask(image).astype(np.uint8)
    # For every pixel: how far is the nearest white line pixel?
    distance_map = cv2.distanceTransform(1 - lines, cv2.DIST_L2, 3)
    distance_map = np.minimum(distance_map, FAR_AWAY)

    def fine_tune(start):
        # Nelder-Mead just tries nearby settings and keeps what helps.
        simplex = start + np.vstack([np.zeros(3), np.diag([0.02, 0.02, 0.05])])
        return minimize(_misfit, start, args=(distance_map,), method="Nelder-Mead",
                        options={"initial_simplex": simplex, "xatol": 1e-5, "fatol": 1e-4, "maxiter": 400})

    if guess is not None:
        found = fine_tune(np.array([guess.pan, guess.tilt, guess.zoom / 1000.0]))
    else:
        # No idea where the camera points. Try lots of settings roughly,
        # then fine tune the 8 most promising ones and keep the winner.
        tries = [(pan, tilt, zoom)
                 for pan in np.radians(np.arange(-48, 49, 3))
                 for tilt in np.radians(np.arange(10, 31, 2))
                 for zoom in np.arange(1.0, 2.31, 0.15)]
        scores = [_misfit(t, distance_map) for t in tries]
        best_few = [np.array(tries[i]) for i in np.argsort(scores)[:8]]
        found = min((fine_tune(start) for start in best_few), key=lambda result: result.fun)
    camera = Camera(found.x[0], found.x[1], found.x[2] * 1000.0)
    return camera, found.fun


def find_players(image, camera):
    """
    Find the players in a picture.
    Returns a list of dicts: {"spot": (x, y) in metres, "kit": name of the kit}.
    """
    # Only look inside the pitch (plus a little margin).
    edge = [(-HALF_LENGTH - 1, -HALF_WIDTH - 1), (HALF_LENGTH + 1, -HALF_WIDTH - 1),
            (HALF_LENGTH + 1, HALF_WIDTH + 1), (-HALF_LENGTH - 1, HALF_WIDTH + 1), (-HALF_LENGTH - 1, -HALF_WIDTH - 1)]
    pixels, in_front = camera.to_pixels(_fine(edge))
    on_pitch = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(on_pitch, [np.round(pixels[in_front]).astype(np.int32)], 1)

    maybe_player = (~grass_mask(image)) & (~white_line_mask(image)) & (on_pitch > 0)

    # Give every non grass pixel the kit colour it is closest to. Doing this
    # per kit means a red player standing in front of a blue one gives two
    # blobs, not one big one.
    kit_names = list(KITS)
    kit_colours = np.array([KITS[k] for k in kit_names], dtype=np.float32)
    gaps = np.linalg.norm(image[:, :, None, :].astype(np.float32) - kit_colours[None, None, :, :], axis=-1)
    nearest_kit = np.argmin(gaps, axis=-1)
    close_enough = gaps.min(axis=-1) < 90

    players = []
    for k, kit in enumerate(kit_names):
        mask = (maybe_player & close_enough & (nearest_kit == k)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, _, boxes, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, count):
            left, top, wide, tall, area = boxes[i]
            if area < 20 or tall < 6:
                continue  # too small to be a player

            # How big SHOULD a player look at this spot in the picture?
            feet_spot = camera.to_pitch([[left + wide / 2.0, top + tall]])[0]
            feet_pixel, _ = camera.to_pixels([feet_spot])
            head_pixel, _ = camera.to_pixels([[feet_spot[0], feet_spot[1], PLAYER_HEIGHT]])
            expected_tall = max(4.0, feet_pixel[0, 1] - head_pixel[0, 1])
            expected_wide = expected_tall * PLAYER_WIDTH / PLAYER_HEIGHT

            # A blob much wider than one player is team mates standing side by side.
            how_many = max(1, int(round(wide / expected_wide - 0.25)))
            for n in range(how_many):
                centre_u = left + wide * (n + 0.5) / how_many
                if tall >= 0.8 * expected_tall:
                    # We can see the whole player, so use the feet.
                    spot = camera.to_pitch([[centre_u, top + tall]])[0]
                else:
                    # Legs are hidden behind someone. Use the head, which is about 1.8 m up.
                    spot = camera.to_pitch([[centre_u, top]], height=PLAYER_HEIGHT)[0]
                if abs(spot[0]) <= HALF_LENGTH + 2 and abs(spot[1]) <= HALF_WIDTH + 2:
                    players.append({"spot": spot, "kit": kit})
    return players


def match_up(found_spots, true_spots, limit=2.5):
    """
    Pair each found player with the nearest real player, one to one.
    Pairs further apart than 'limit' metres don't count.
    Returns a list of (found index, true index, distance).
    """
    if len(found_spots) == 0 or len(true_spots) == 0:
        return []
    gaps = np.linalg.norm(np.asarray(found_spots)[:, None, :] - np.asarray(true_spots)[None, :, :], axis=-1)
    return [(r, c, gaps[r, c]) for r, c in match_up_table(gaps, limit)]


def match_up_table(gaps, limit=2.5):
    """Same as match_up, but you hand in the table of distances yourself. Returns (row, column) pairs."""
    rows, cols = linear_sum_assignment(gaps)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if gaps[r, c] <= limit]


def follow_players(found_per_frame, seconds_per_frame, reach=3.0):
    """
    Join the dots between frames, so we know that the player found here
    in frame 5 is the same player found there in frame 4.

    For each kit we guess where every known player will be next (they
    keep moving the way they were), then pair guesses with what we found.
    Returns a list of tracks: {"kit": name, "path": (frames, 2) array}.
    Frames where the player was not seen hold NaN.
    """
    frames = len(found_per_frame)
    tracks = []
    for t, found in enumerate(found_per_frame):
        for kit in KITS:
            spots = [p["spot"] for p in found if p["kit"] == kit]
            live = [k for k in tracks if k["kit"] == kit and t - k["seen"] <= 5]
            guesses = [k["last"] + k["speed"] * (t - k["seen"]) * seconds_per_frame for k in live]
            pairs = match_up(spots, guesses, limit=reach)
            taken = set()
            for spot_index, track_index, _ in pairs:
                track, spot = live[track_index], spots[spot_index]
                waited = (t - track["seen"]) * seconds_per_frame
                track["speed"] = 0.5 * track["speed"] + 0.5 * (spot - track["last"]) / waited
                track["last"], track["seen"] = spot, t
                track["path"][t] = spot
                taken.add(spot_index)
            for i, spot in enumerate(spots):
                if i not in taken:
                    path = np.full((frames, 2), np.nan)
                    path[t] = spot
                    tracks.append({"kit": kit, "path": path, "last": spot, "seen": t, "speed": np.zeros(2)})
    return [{"kit": k["kit"], "path": k["path"]} for k in tracks]


def fill_gaps(path):
    """
    Fill in frames where a player was hidden, by drawing a straight line
    between the last and next place we saw them. Before the first sighting
    and after the last one we just hold the nearest known spot.
    """
    seen = ~np.isnan(path[:, 0])
    where = np.arange(len(path))
    return np.stack([np.interp(where, where[seen], path[seen, axis]) for axis in (0, 1)], axis=1)
