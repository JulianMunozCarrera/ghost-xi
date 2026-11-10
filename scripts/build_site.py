"""
Builds the website (docs/index.html) as one self-contained file.

It packs together:
  - the trained model weights
  - the 19 goals tracked from TV
  - the result numbers from the results folder
  - the JavaScript engine and the page itself (from the web folder)

It also writes web/expected.json: answers worked out in Python for a
few moments. A test runs the JavaScript engine on the same moments and
checks that it gets the same answers.

    python scripts/build_site.py --lastrow data/lastrow/datasets/positional_data/liverpool_2019.csv
"""

import argparse
import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ghostxi.analysis import analyse_shot, start_frame
from ghostxi.data_lastrow import load_goals
from ghostxi.ghost import make_ghosts
from ghostxi.model_numpy import GhostModel

ROOT = os.path.join(os.path.dirname(__file__), "..")


def pack_weights(path):
    """All weights as one base64 string, plus a list saying what is where."""
    weights = np.load(path)
    names = sorted(weights.files)
    manifest = [[name, list(weights[name].shape)] for name in names]
    raw = b"".join(weights[name].astype("<f4").tobytes() for name in names)
    return {"manifest": manifest, "data": base64.b64encode(raw).decode("ascii")}


def site_clip(clip):
    """Cut a clip down to what the website shows: one second of lead-in, then the 8 seconds before the shot, then the goal."""
    first = max(0, start_frame(clip) - int(clip.fps))
    return clip.cut(first, len(clip.ball) - 1)


def pack_clip(clip):
    """A clip as plain lists of whole centimeters."""
    def cm(a):
        return np.round(a.astype(np.float64) * 100).astype(int).ravel().tolist()
    return {
        "name": clip.name, "fps": clip.fps, "frames": int(len(clip.ball)), "shot": int(clip.shot_frame),
        "nAtt": int(clip.attackers.shape[1]), "nDef": int(clip.defenders.shape[1]),
        "attKeeper": int(clip.att_keeper), "defKeeper": int(clip.def_keeper),
        "ball": cm(clip.ball), "att": cm(clip.attackers), "def": cm(clip.defenders),
    }


def expected_answers(model, clips):
    """Python's answers for a few moments, for the JavaScript test to compare against."""
    out = []
    for index in (0, 8, 16):
        clip = clips[index]
        start = start_frame(clip)
        frames = [start + 1, (start + clip.shot_frame) // 2, clip.shot_frame]
        team = make_ghosts(model, clip, start, solo=False, only_frames=frames)
        solo = make_ghosts(model, clip, start, solo=True, only_frames=frames)
        shot = analyse_shot(model, clip)
        out.append({
            "clip": index, "start": int(start), "frames": [int(f) for f in frames],
            "team": [team[f].ravel().tolist() for f in frames],
            "solo": [solo[f].ravel().tolist() for f in frames],
            "shot": {k: shot[k] for k in ("off_by", "danger_real", "danger_ghost", "space_real", "space_ghost")},
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lastrow", required=True)
    parser.add_argument("--weights", default=os.path.join(ROOT, "models", "ghost.npz"))
    parser.add_argument("--out", default=os.path.join(ROOT, "docs", "index.html"))
    args = parser.parse_args()

    model = GhostModel(args.weights)
    clips = [site_clip(c) for c in load_goals(args.lastrow)]

    def read_json(name):
        path = os.path.join(ROOT, "results", name)
        return json.load(open(path)) if os.path.exists(path) else None

    data = {
        "weights": pack_weights(args.weights),
        "clips": [pack_clip(c) for c in clips],
        "results": {"model": read_json("model_results.json"), "goals": read_json("goals.json"),
                    "camera": read_json("camera_test.json")},
    }
    with open(os.path.join(ROOT, "web", "expected.json"), "w") as f:
        json.dump(expected_answers(model, clips), f)
    with open(os.path.join(ROOT, "web", "data.json"), "w") as f:
        json.dump(data, f, separators=(",", ":"))

    page_path = os.path.join(ROOT, "web", "page.html")
    if os.path.exists(page_path):
        page = open(page_path).read()
        page = page.replace("/*__ENGINE__*/", open(os.path.join(ROOT, "web", "engine.js")).read())
        page = page.replace("/*__APP__*/", open(os.path.join(ROOT, "web", "app.js")).read())
        page = page.replace("/*__DATA__*/", "window.GHOST_DATA = " + json.dumps(data, separators=(",", ":")) + ";")
        with open(args.out, "w") as f:
            f.write(page)
        print("wrote", args.out, f"({len(page) / 1e6:.2f} MB)")
    print("clips:", len(clips), "weights:", f"{len(data['weights']['data']) / 1e6:.2f} MB of base64")


if __name__ == "__main__":
    main()
