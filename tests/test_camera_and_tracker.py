import numpy as np

from ghostxi.camera import Camera, pitch_landmarks, find_homography, find_homography_robust, apply_homography
from ghostxi.tracker import Tracker


def test_camera_round_trip_is_exact_with_perfect_pixels():
    camera = Camera()
    camera.aim = np.array([30.0, 5.0])
    landmarks = pitch_landmarks()
    pixels, inside = camera.project(landmarks)
    assert inside.sum() >= 4
    table = find_homography(landmarks[inside], pixels[inside])

    players = np.random.default_rng(0).uniform([5, -25], [50, 25], (50, 2))
    player_pixels, seen = camera.project(players)
    back = apply_homography(np.linalg.inv(table), player_pixels[seen])
    assert np.abs(back - players[seen]).max() < 1e-6


def test_robust_fit_survives_wrong_landmarks():
    rng = np.random.default_rng(0)
    camera = Camera()
    camera.aim = np.array([35.0, 0.0])
    landmarks = pitch_landmarks()
    pixels, inside = camera.project(landmarks)
    found = pixels[inside] + rng.normal(0, 1.0, (inside.sum(), 2))
    found[:2] = rng.uniform(0, 700, (2, 2))   # two landmarks are plain wrong
    table, trusted = find_homography_robust(landmarks[inside], found, rng)
    assert table is not None
    assert not trusted[:2].any()              # the wrong ones were thrown out
    spot = np.array([[41.5, 0.0]])            # the penalty spot
    back = apply_homography(np.linalg.inv(table), camera.project(spot)[0])
    assert np.linalg.norm(back - spot) < 0.5


def test_robust_fit_gives_up_with_too_few_landmarks():
    table, trusted = find_homography_robust(np.zeros((3, 2)), np.zeros((3, 2)), np.random.default_rng(0))
    assert table is None and trusted is None


def test_far_side_of_the_pitch_is_smaller_in_the_picture():
    camera = Camera()
    near, _ = camera.project(np.array([[-5.0, -30.0], [5.0, -30.0]]))
    far, _ = camera.project(np.array([[-5.0, 30.0], [5.0, 30.0]]))
    assert abs(near[1, 0] - near[0, 0]) > abs(far[1, 0] - far[0, 0])


def test_tracker_keeps_numbers_when_two_players_run_side_by_side():
    tracker = Tracker(fps=10)
    numbers = []
    for step in range(30):
        spots = np.array([[step * 0.5, 0.0], [step * 0.5, 4.0]])
        numbers.append(tracker.update(spots[::-1] if step % 2 else spots))   # shuffle the order each frame
    first = numbers[0]
    for step, given in enumerate(numbers):
        assert (given[::-1] if step % 2 else given) == first


def test_tracker_gives_a_new_number_after_a_long_gap():
    tracker = Tracker(fps=10, forget_after=1.0)
    first = tracker.update(np.array([[0.0, 0.0]]))[0]
    for _ in range(15):                       # gone for 1.5 seconds
        tracker.update(np.zeros((0, 2)))
    assert tracker.update(np.array([[0.0, 0.0]]))[0] != first


def test_tracker_keeps_the_number_after_a_short_gap():
    tracker = Tracker(fps=10, forget_after=1.0)
    first = tracker.update(np.array([[0.0, 0.0]]))[0]
    for _ in range(3):
        tracker.update(np.zeros((0, 2)))
    assert tracker.update(np.array([[0.3, 0.0]]))[0] == first
