"""
Runs the ghosts on real goals and writes down what they show.

Three groups of moments, all from matches the model did not train on:
  - ordinary moments: the ball is in the attacking third, nothing special
  - shots that did not go in
  - goals

Plus the 19 Liverpool goals tracked from TV (a different data source).

If the ghosts mean anything, real defenders should be further from
their ghosts before a goal than at an ordinary moment.

    python scripts/analyse_goals.py --metrica data/metrica/data --lastrow data/lastrow/datasets/positional_data/liverpool_2019.csv
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ghostxi.analysis import analyse_shot, LOOK_BACK
from ghostxi.clips import clip_from_match
from ghostxi.data_lastrow import load_goals
from ghostxi.data_metrica import load_all
from ghostxi.model_numpy import GhostModel
from ghostxi.samples import MatchSamples

MIN_SECONDS = 3.0   # skip shots with less than this much live play before them (free kicks, penalties)


def outfield_off_by(result, clip):
    return [v for i, v in enumerate(result["off_by"]) if i != clip.def_keeper]


def summarise(rows):
    """Average the numbers of a group of moments. Also says how sure we are (see spread)."""
    if not rows:
        return {"count": 0}
    rng = np.random.default_rng(0)

    def mean_and_range(values):
        values = np.array(values, dtype=np.float64)
        # Bootstrap: redo the average many times on random re-draws of the
        # same moments. The middle 95% of those averages is our range.
        redraws = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(2000)]
        return {"mean": round(float(values.mean()), 3),
                "low": round(float(np.percentile(redraws, 2.5)), 3),
                "high": round(float(np.percentile(redraws, 97.5)), 3)}

    return {
        "count": len(rows),
        "off_by_m": mean_and_range([r["mean_off_by"] for r in rows]),
        "worst_off_by_m": mean_and_range([r["worst_off_by"] for r in rows]),
        "danger_real": mean_and_range([r["danger_real"] for r in rows]),
        "danger_ghost": mean_and_range([r["danger_ghost"] for r in rows]),
        "danger_real_minus_ghost": mean_and_range([r["danger_real"] - r["danger_ghost"] for r in rows]),
        "space_real_m": mean_and_range([r["space_real"] for r in rows]),
        "space_ghost_m": mean_and_range([r["space_ghost"] for r in rows]),
    }


def flatten(result, clip, name, kind):
    off = outfield_off_by(result, clip)
    return {
        "name": name, "kind": kind, "source": clip.source, "seconds": round(result["seconds"], 2),
        "mean_off_by": float(np.mean(off)), "worst_off_by": float(np.max(off)),
        "worst_defender": int(np.argmax([v if i != clip.def_keeper else -1 for i, v in enumerate(result["off_by"])])),
        "off_by": [round(v, 2) for v in result["off_by"]],
        "danger_real": result["danger_real"], "danger_ghost": result["danger_ghost"],
        "space_real": result["space_real"], "space_ghost": result["space_ghost"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrica", required=True)
    parser.add_argument("--lastrow", required=True)
    parser.add_argument("--models", default="models")
    parser.add_argument("--out", default="results/goals.json")
    args = parser.parse_args()

    rows = []
    matches = load_all(args.metrica)
    for number, match in enumerate(matches, start=1):
        # Each match is analysed by the model that did NOT train on it.
        model = GhostModel(os.path.join(args.models, f"ghost_round{number}.npz"))
        samples = MatchSamples(match)
        longest = dict(zip(samples.times.tolist(), samples.longest.tolist()))
        alive = ~np.isnan(match.ball[:, 0])

        def live_frames_before(frame):
            count = 0
            while frame - count - 1 >= 0 and alive[frame - count - 1] and count < 10 * match.fps:
                count += 1
            return count

        goal_number = 0
        shot_frames = set()
        for shot in match.shots:
            if shot["penalty"]:
                continue
            back = min(live_frames_before(shot["frame"]), int(LOOK_BACK * match.fps))
            if back < MIN_SECONDS * match.fps or match.possession[shot["frame"]] != shot["team"]:
                continue
            if shot["goal"]:
                goal_number += 1
            name = f"{match.name}, goal {goal_number}" if shot["goal"] else f"{match.name}, shot at frame {shot['frame']}"
            clip = clip_from_match(match, shot["frame"] - back, shot["frame"], name)
            rows.append(flatten(analyse_shot(model, clip), clip, name, "goal" if shot["goal"] else "shot"))
            shot_frames.add(shot["frame"])

        # Ordinary moments: ball in the attacking third, 8 seconds of live play before, one every 2 seconds.
        for frame, back in longest.items():
            if frame % (2 * match.fps) != 0 or back < LOOK_BACK * match.fps:
                continue
            if any(abs(frame - s) < 10 * match.fps for s in shot_frames):
                continue  # keep these separate from the shots
            sign = match.direction[(int(match.period[frame]), int(match.possession[frame]))]
            if match.ball[frame, 0] * sign < 17.5:
                continue
            clip = clip_from_match(match, frame - int(LOOK_BACK * match.fps), frame, "ordinary")
            rows.append(flatten(analyse_shot(model, clip), clip, f"{match.name}, frame {frame}", "ordinary"))
        print(match.name, "done:", {k: sum(r["kind"] == k and r["source"] == match.name for r in rows)
                                    for k in ("ordinary", "shot", "goal")}, flush=True)

    # The TV goals use the final model. It trained on Metrica only, so these are new to it.
    final = GhostModel(os.path.join(args.models, "ghost.npz"))
    for clip in load_goals(args.lastrow):
        rows.append(flatten(analyse_shot(final, clip), clip, clip.name, "tv_goal"))

    report = {
        "groups": {kind: summarise([r for r in rows if r["kind"] == kind])
                   for kind in ("ordinary", "shot", "goal", "tv_goal")},
        "goals": [r for r in rows if r["kind"] in ("goal", "tv_goal")],
        "shots": [r for r in rows if r["kind"] == "shot"],
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    for kind, group in report["groups"].items():
        print(kind, json.dumps(group))


if __name__ == "__main__":
    main()
