"""
Tests the "TV picture to pitch positions" stage with a pretend camera.

We do not have TV footage we are allowed to share, so we fake the
camera. We take a real match where we know exactly where everyone is,
point a pretend TV camera at it, and add the kinds of mistakes a real
system makes:
  - landmarks are found a few pixels off, and 1 in 10 is plain wrong
  - players are found a few pixels off, and 1 in 20 is missed
  - players outside the picture are not seen at all

Then we rebuild the pitch positions from the pixels alone and check
them against the truth. Last, we feed the rebuilt positions to the
ghost model to see how much the camera hurts it.

What this does NOT test: finding players and landmarks in real pixels.
That needs a trained detector and real footage.

    python scripts/camera_test.py --data data/metrica/data
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ghostxi.camera import Camera, pitch_landmarks, find_homography_robust, apply_homography
from ghostxi.data_metrica import load_epts_match
from ghostxi.model_numpy import GhostModel
from ghostxi.pitch import speed_from_positions
from ghostxi.tokens import build_tokens, moves_to_positions
from ghostxi.tracker import Tracker

SKIP = 2                 # use every 2nd frame, so 12.5 pictures per second
LANDMARK_NOISE = 2.0     # pixels
PLAYER_NOISE = 3.0       # pixels
WRONG_LANDMARKS = 0.10
MISSED_PLAYERS = 0.05
GAP_SECONDS = 3.0        # "then" is this long before "now" in the ghost test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--minutes", type=float, default=15.0)
    parser.add_argument("--model", default="models/ghost_round3.npz")
    parser.add_argument("--out", default="results/camera_test.json")
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    match = load_epts_match(os.path.join(args.data, "Sample_Game_3"))
    fps = match.fps / SKIP
    alive = np.where(~np.isnan(match.ball[:, 0]) & (match.possession >= 0))[0]
    frames = alive[alive % SKIP == 0][:int(args.minutes * 60 * fps)]

    landmarks = pitch_landmarks()
    camera = Camera()
    trackers = [Tracker(fps), Tracker(fps)]

    raw_errors, errors, in_view, calibrated = [], [], [], 0
    seen_as = {}       # frame -> {(team, slot): (track number, smoothed position, smoothed speed)}
    ball_seen = {}     # frame -> rebuilt ball position

    for frame in frames:
        camera.follow(match.ball[frame])
        seen_as[frame] = {}

        # 1. Landmarks in the picture, with mistakes.
        pixels, inside = camera.project(landmarks)
        found = pixels[inside] + rng.normal(0, LANDMARK_NOISE, (inside.sum(), 2))
        wrong = rng.random(inside.sum()) < WRONG_LANDMARKS
        found[wrong] = rng.uniform([0, 0], [camera.WIDTH, camera.HEIGHT], (wrong.sum(), 2))
        table, _ = find_homography_robust(landmarks[inside], found, rng)
        if table is None:
            for tracker in trackers:
                tracker.update(np.zeros((0, 2)))   # nothing seen this frame
            continue
        calibrated += 1
        to_pitch = np.linalg.inv(table)

        # 2. Players in the picture, with mistakes.
        count_in_view = 0
        for team in (0, 1):
            slots = np.where(~np.isnan(match.xy[frame, team, :, 0]))[0]
            truth = match.xy[frame, team, slots].astype(float)
            pixels, inside = camera.project(truth)
            detected = inside & (rng.random(len(slots)) >= MISSED_PLAYERS)
            count_in_view += inside.sum()
            rebuilt = apply_homography(to_pitch, pixels[detected] + rng.normal(0, PLAYER_NOISE, (detected.sum(), 2)))
            raw_errors.extend(np.linalg.norm(rebuilt - truth[detected], axis=1))

            # 3. Link the spots over time. The tracker also smooths them.
            numbers = trackers[team].update(rebuilt)
            for slot, number, real in zip(slots[detected], numbers, truth[detected]):
                track = trackers[team].tracks[number]
                errors.append(np.linalg.norm(track["position"] - real))
                seen_as[frame][(team, int(slot))] = (number, track["position"].copy(), track["velocity"].copy())
        in_view.append(count_in_view)

        pixels, inside = camera.project(match.ball[frame][None].astype(float))
        if inside[0]:
            ball_seen[frame] = apply_homography(to_pitch, pixels + rng.normal(0, PLAYER_NOISE, (1, 2)))[0]

    report = {
        "minutes_of_play": round(len(frames) / fps / 60, 1),
        "pictures": int(len(frames)),
        "pictures_we_could_map": round(calibrated / len(frames), 3),
        "players_in_picture_out_of_22": round(float(np.mean(in_view)), 1),
        "position_error_m_before_smoothing": {"typical": round(float(np.median(raw_errors)), 2),
                                              "bad_case_95": round(float(np.percentile(raw_errors, 95)), 2)},
        "position_error_m": {"typical": round(float(np.median(errors)), 2),
                             "bad_case_95": round(float(np.percentile(errors, 95)), 2)},
    }

    # ---- Ghost test: does the model still work on camera data?
    model = GhostModel(args.model)
    true_speed = speed_from_positions(match.xy, match.fps)
    true_ball_speed = speed_from_positions(match.ball, match.fps, top_speed=40.0)
    gap = int(GAP_SECONDS * match.fps) // SKIP * SKIP   # must land on a frame we kept

    def camera_ball_speed(frame):
        for steps in (4, 2):
            earlier = ball_seen.get(frame - steps * SKIP)
            if earlier is not None:
                return (ball_seen[frame] - earlier) / (steps * SKIP / match.fps)
        return np.zeros(2)

    cam_vs_truth, full_vs_truth, cam_vs_full, could_ghost, defenders_total = [], [], [], 0, 0
    seen_both, same_number = 0, 0
    for now in frames[::int(2 * fps)]:
        then = now - gap
        if then not in seen_as or now not in ball_seen or then not in ball_seen or not seen_as[now]:
            continue
        attacking = int(match.possession[now])
        defending = 1 - attacking
        period = int(match.period[now])
        if match.period[then] != period or np.isnan(match.ball[then, 0]):
            continue
        sign = match.direction[(period, attacking)]

        def camera_moment(frame):
            return {"ball": ball_seen[frame] * sign, "ball_v": camera_ball_speed(frame) * sign,
                    "att_keeper": -1, "def_keeper": -1}

        cam_now, cam_then = camera_moment(now), camera_moment(then)
        full_now = {"ball": match.ball[now] * sign, "ball_v": true_ball_speed[now] * sign}
        full_then = {"ball": match.ball[then] * sign, "ball_v": true_ball_speed[then] * sign}
        truth_now = {}
        for key_name, team in (("att", attacking), ("def", defending)):
            slots = np.where(~np.isnan(match.xy[now, team, :, 0]) & ~np.isnan(match.xy[then, team, :, 0]))[0]
            keeper_slot = match.keeper[(period, team)]
            keeper = int(np.where(slots == keeper_slot)[0][0]) if keeper_slot in slots else -1
            n = len(slots)
            for moment in (cam_now, cam_then):
                moment[key_name] = np.full((n, 2), np.nan)
                moment[key_name + "_v"] = np.zeros((n, 2))
                moment[key_name + "_keeper"] = keeper
            for i, slot in enumerate(slots):
                key = (team, int(slot))
                a, b = seen_as[now].get(key), seen_as[then].get(key)
                if a is not None and b is not None:
                    seen_both += 1
                    same_number += int(a[0] == b[0])
                # A defender only needs to be seen "then". Everyone else needs both, with the same number.
                if b is not None:
                    cam_then[key_name][i] = b[1] * sign
                    cam_then[key_name + "_v"][i] = b[2] * sign
                if a is not None and b is not None and a[0] == b[0]:
                    cam_now[key_name][i] = a[1] * sign
                    cam_now[key_name + "_v"][i] = a[2] * sign
                elif key_name == "att":
                    cam_then[key_name][i] = np.nan
            for moment, frame in ((full_now, now), (full_then, then)):
                moment[key_name] = match.xy[frame, team, slots] * sign
                moment[key_name + "_v"] = true_speed[frame, team, slots] * sign
                moment[key_name + "_keeper"] = keeper
            truth_now[key_name] = match.xy[now, team, slots] * sign

        n_def = len(truth_now["def"])
        hide = [True] * n_def
        seconds = gap / match.fps
        tokens, present = build_tokens(cam_now, cam_then, seconds, hide)
        cam_ghost = moves_to_positions(model.run(tokens, present), np.nan_to_num(cam_then["def"]))
        tokens, present = build_tokens(full_now, full_then, seconds, hide)
        full_ghost = moves_to_positions(model.run(tokens, present), full_then["def"])

        seen_then = ~np.isnan(cam_then["def"][:, 0])
        defenders_total += n_def
        could_ghost += int(seen_then.sum())
        cam_vs_truth.extend(np.linalg.norm(cam_ghost - truth_now["def"], axis=1)[seen_then])
        full_vs_truth.extend(np.linalg.norm(full_ghost - truth_now["def"], axis=1)[seen_then])
        cam_vs_full.extend(np.linalg.norm(cam_ghost - full_ghost, axis=1)[seen_then])

    report["ghost_test"] = {
        "seconds_between_then_and_now": round(gap / match.fps, 2),
        "players_in_picture_at_both_moments_who_kept_their_number": round(same_number / max(seen_both, 1), 3),
        "defenders_we_could_ghost": round(could_ghost / max(defenders_total, 1), 3),
        "ghost_miss_m_with_perfect_data": round(float(np.mean(full_vs_truth)), 2),
        "ghost_miss_m_with_camera_data": round(float(np.mean(cam_vs_truth)), 2),
        "ghost_moved_by_m_because_of_camera": round(float(np.mean(cam_vs_full)), 2),
        "defenders_checked": len(cam_vs_truth),
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
