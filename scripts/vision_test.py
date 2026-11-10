"""
Tests the video part of the project and prints a report.

We can't use real match video (it is owned by the leagues), so we draw
our own from real tracking data. See ghostxi/render.py. Because we drew
it, we know the true answer for every frame.

For the play before each shot in the three matches we:
  1. draw what the TV camera would see, 5 pictures per second,
  2. find the camera and the players from the pictures alone,
  3. compare with the truth.

Then, for the goals, we run the ghost model on the positions found
from video and compare with the ghosts made from the true positions.

    python scripts/vision_test.py --data data/metrica/data
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ghostxi import render, vision
from ghostxi.analysis import clip_around_shot, as_batch
from ghostxi.camera import IMAGE_WIDTH, IMAGE_HEIGHT
from ghostxi.data import load_all, shots, STEP
from ghostxi.model import load, ghost_paths


def watch_clip(spell):
    """Draw every frame of a clip, then find the camera and players in each one."""
    on, att_on = spell.defender_on, spell.attacker_on
    keeper = int(np.where(spell.keeper[on])[0][0]) if spell.keeper[on].any() else None
    true_cameras = render.follow_ball(spell.ball)
    guess, frames = None, []
    for t, true_camera in enumerate(true_cameras):
        defenders, attackers = spell.defenders[t][on], spell.attackers[t][att_on]
        image = render.draw_frame(true_camera, defenders, attackers, spell.ball[t], keeper, None, seed=t)
        camera, fit = vision.find_camera(image, guess)
        guess = camera
        frames.append({"image": image, "true_camera": true_camera, "camera": camera,
                       "found": vision.find_players(image, camera),
                       "defenders": defenders, "attackers": attackers, "keeper": keeper})
    return frames


def in_view(camera, spots, margin=8):
    pixels, in_front = camera.to_pixels(spots)
    return (in_front & (pixels[:, 0] > margin) & (pixels[:, 0] < IMAGE_WIDTH - margin)
            & (pixels[:, 1] > margin) & (pixels[:, 1] < IMAGE_HEIGHT - margin))


def score_frames(frames, totals):
    """Add this clip's numbers to the running totals."""
    check_spots = np.array([(x, y) for x in range(-50, 51, 10) for y in range(-30, 31, 10)], dtype=float)
    for frame in frames:
        totals["frames"] += 1
        # Camera: take pitch spots the camera can see, send them to the picture with the
        # TRUE camera and back to the pitch with the FOUND camera. A perfect answer gives 0 m.
        seen = in_view(frame["true_camera"], check_spots)
        pixels, _ = frame["true_camera"].to_pixels(check_spots[seen])
        totals["camera_miss"] += list(np.linalg.norm(frame["camera"].to_pitch(pixels) - check_spots[seen], axis=1))

        # Players: pair what we found with who is really there (and in view).
        truth = np.concatenate([frame["defenders"], frame["attackers"]])
        true_kit = ["defender"] * len(frame["defenders"]) + ["attacker"] * len(frame["attackers"])
        if frame["keeper"] is not None:
            true_kit[frame["keeper"]] = "defender_keeper"
        visible = np.where(in_view(frame["true_camera"], truth))[0]
        pairs = vision.match_up([p["spot"] for p in frame["found"]], truth[visible])
        totals["players_in_view"] += len(visible)
        totals["players_found"] += len(pairs)
        totals["things_found"] += len(frame["found"])
        totals["position_miss"] += [gap for _, _, gap in pairs]
        totals["kit_right"] += sum(frame["found"][f]["kit"] == true_kit[visible[v]] for f, v, _ in pairs)


def ghosts_from_video(frames, spell, net):
    """
    Build the model's input from what vision found, run the ghosts, and
    compare with ghosts made from the true positions.
    The ball comes from the tracking data. Finding a ball this small in
    a picture is its own hard problem and is not done yet.
    """
    tracks = vision.follow_players([f["found"] for f in frames], STEP)
    solid = [k for k in tracks if (~np.isnan(k["path"][:, 0])).mean() >= 0.85]
    defenders = [vision.fill_gaps(k["path"]) for k in solid if k["kit"].startswith("defender")][:11]
    keeper_flags = [k["kit"] == "defender_keeper" for k in solid if k["kit"].startswith("defender")][:11]
    attackers = [vision.fill_gaps(k["path"]) for k in solid if k["kit"].startswith("attacker")][:11]
    if len(defenders) < 4 or len(attackers) < 4:
        return None

    def pad(paths):
        out = np.zeros((len(frames), 11, 2), dtype=np.float32)
        mask = np.zeros(11, dtype=bool)
        for i, path in enumerate(paths):
            out[:, i], mask[i] = path, True
        return out, mask

    seen_defenders, defender_on = pad(defenders)
    seen_attackers, attacker_on = pad(attackers)
    keeper = np.zeros(11, dtype=bool)
    keeper[:len(keeper_flags)] = keeper_flags
    from_video = ghost_paths(net, {
        "defenders": seen_defenders[None], "attackers": seen_attackers[None], "ball": spell.ball[None],
        "defender_on": defender_on[None], "attacker_on": attacker_on[None], "keeper": keeper[None]})[0]
    from_truth = ghost_paths(net, as_batch(spell))[0]

    # Work out which true defender each video track belongs to, then compare their ghosts.
    true_on = np.where(spell.defender_on)[0]
    typical_gap = np.array([[np.linalg.norm(path - spell.defenders[:, j], axis=1).mean() for j in true_on] for path in defenders])
    pairs = vision.match_up_table(typical_gap, limit=2.5)
    if not pairs:
        return None
    ghost_gap = np.mean([np.linalg.norm(from_video[:, i] - from_truth[:, true_on[j]], axis=1).mean() for i, j in pairs])
    return {"defenders_tracked": len(pairs), "defenders_real": int(len(true_on)),
            "attackers_tracked": len(attackers), "ghost_gap_m": float(ghost_gap)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--models", default="models")
    parser.add_argument("--out", default="results_vision.json")
    parser.add_argument("--save-picture", default="web/broadcast_frame.jpg")
    args = parser.parse_args()

    totals = {"camera_miss": [], "position_miss": [], "players_in_view": 0, "players_found": 0,
              "things_found": 0, "kit_right": 0, "frames": 0}
    end_to_end, clips = [], 0
    for match in load_all(args.data):
        net = load(os.path.join(args.models, f"ghost_without_{match.name}.pt"))
        for shot in shots(match):
            spell, shot_index = clip_around_shot(match, shot)
            if spell is None:
                continue
            frames = watch_clip(spell)
            score_frames(frames, totals)
            clips += 1
            if shot["goal"]:
                result = ghosts_from_video(frames, spell, net)
                if result:
                    end_to_end.append(result)
                if args.save_picture and not os.path.exists(args.save_picture):
                    cv2.imwrite(args.save_picture, frames[shot_index - 3]["image"], [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"  {match.name} shot at frame {shot['frame']}: done ({clips} clips so far)", flush=True)

    report = {
        "clips": clips,
        "frames": totals["frames"],
        "camera_miss_m_typical": round(float(np.median(totals["camera_miss"])), 3),
        "camera_miss_m_worst_5_percent": round(float(np.percentile(totals["camera_miss"], 95)), 3),
        "players_found_share": round(totals["players_found"] / totals["players_in_view"], 3),
        "found_things_that_are_players_share": round(totals["players_found"] / totals["things_found"], 3),
        "position_miss_m_typical": round(float(np.median(totals["position_miss"])), 3),
        "position_miss_m_worst_5_percent": round(float(np.percentile(totals["position_miss"], 95)), 3),
        "kit_right_share": round(totals["kit_right"] / totals["players_found"], 3),
        "goals_run_end_to_end": len(end_to_end),
        "end_to_end_defenders_tracked": int(sum(r["defenders_tracked"] for r in end_to_end)),
        "end_to_end_defenders_real": int(sum(r["defenders_real"] for r in end_to_end)),
        "end_to_end_ghost_gap_m": round(float(np.mean([r["ghost_gap_m"] for r in end_to_end])), 3) if end_to_end else None,
    }
    print(json.dumps(report, indent=1))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
