"""
Runs the ghosts on real shots and goals and writes down what they show.

Three groups of moments, all from matches the model did not train on:
  - ordinary moments: the ball is in the attacking third, nothing special
  - shots that did not go in
  - goals

Plus the 19 Liverpool goals tracked from TV by Last Row. That is a
third data source the model never trained on at all.

If the ghosts mean anything, real defenders should be further from
their ghosts before a goal than at an ordinary moment.

    python scripts/analyse_goals.py --metrica data/metrica/data --skillcorner data/skillcorner/data \\
        --lastrow data/lastrow/datasets/positional_data/liverpool_2019.csv
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
from ghostxi import data_metrica, data_skillcorner
from ghostxi.model_numpy import GhostModel
from ghostxi.samples import MatchSamples

MIN_SECONDS = 3.0   # skip shots with less than this much live play before them (free kicks, penalties)
MIN_COMPARED = 3    # skip moments where fewer outfield defenders than this can be compared (TV shows too few)


def outfield_off_by(result, clip):
    """Off-by distances of the outfield defenders we could compare (no goalkeeper, no NaN)."""
    return [v for i, v in enumerate(result["off_by"]) if i != clip.def_keeper and np.isfinite(v)]


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


def flatten(result, clip, name, kind, source):
    off = outfield_off_by(result, clip)
    return {
        "name": name, "kind": kind, "source": source, "seconds": round(result["seconds"], 2),
        "compared": len(off),
        "mean_off_by": float(np.mean(off)), "worst_off_by": float(np.max(off)),
        # NaN (could not compare) is written as null, so the file stays valid JSON.
        "off_by": [round(v, 2) if np.isfinite(v) else None for v in result["off_by"]],
        "danger_real": result["danger_real"], "danger_ghost": result["danger_ghost"],
        "space_real": result["space_real"], "space_ghost": result["space_ghost"],
    }


def study_match(match, model, rows):
    """Add the shots, goals and ordinary moments of one match to rows."""
    source = "stadium" if match.source == "metrica" else "tv"
    samples = MatchSamples(match)
    longest = dict(zip(samples.times.tolist(), samples.longest.tolist()))
    alive = ~np.isnan(match.ball[:, 0])
    before = len(rows)

    def live_frames_before(frame):
        count = 0
        while frame - count - 1 >= 0 and alive[frame - count - 1] and count < 10 * match.fps:
            count += 1
        return count

    def add(clip, name, kind):
        result = analyse_shot(model, clip)
        if len(outfield_off_by(result, clip)) >= MIN_COMPARED and np.isfinite(result["space_real"]):
            rows.append(flatten(result, clip, name, kind, source))

    goal_number = 0
    shot_frames = set()
    for shot in match.shots:
        if shot["penalty"]:
            continue
        back = min(live_frames_before(shot["frame"]), int(LOOK_BACK * match.fps))
        if back < MIN_SECONDS * match.fps or match.possession[shot["frame"]] != shot["team"]:
            continue
        goal_number += int(shot["goal"])
        name = f"{match.name}, goal {goal_number}" if shot["goal"] else f"{match.name}, shot at frame {shot['frame']}"
        add(clip_from_match(match, shot["frame"] - back, shot["frame"], name), name, "goal" if shot["goal"] else "shot")
        shot_frames.add(shot["frame"])

    # Ordinary moments: ball in the attacking third, 8 seconds of live play before, one every 2 seconds.
    for frame, back in longest.items():
        if frame % int(2 * match.fps) != 0 or back < LOOK_BACK * match.fps:
            continue
        if any(abs(frame - s) < 10 * match.fps for s in shot_frames):
            continue  # keep these separate from the shots
        sign = match.direction[(int(match.period[frame]), int(match.possession[frame]))]
        if match.ball[frame, 0] * sign < 17.5:
            continue
        clip = clip_from_match(match, frame - int(LOOK_BACK * match.fps), frame, "ordinary")
        add(clip, f"{match.name}, frame {frame}", "ordinary")
    print(match.name, "done:", {k: sum(r["kind"] == k for r in rows[before:]) for k in ("ordinary", "shot", "goal")}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrica", required=True)
    parser.add_argument("--skillcorner", required=True)
    parser.add_argument("--lastrow", required=True)
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--models", default="models")
    parser.add_argument("--model-results", default="results/model_results.json")
    parser.add_argument("--out", default="results/goals.json")
    args = parser.parse_args()

    # Which round left each match out? That round's model is the one that never saw it.
    tested_in_round = json.load(open(args.model_results))["tested_in_round"]
    models = {}
    rows = []
    matches = data_metrica.load_all(args.metrica) + data_skillcorner.load_all(args.skillcorner, args.cache)
    for match in matches:
        number = tested_in_round[match.name]
        if number not in models:
            models[number] = GhostModel(os.path.join(args.models, f"ghost_round{number}.npz"))
        study_match(match, models[number], rows)

    # The Liverpool goals use the final model. They come from a different data source, so they are new to it.
    final = GhostModel(os.path.join(args.models, "ghost.npz"))
    for clip in load_goals(args.lastrow):
        rows.append(flatten(analyse_shot(final, clip), clip, clip.name, "tv_goal", "last_row"))

    report = {
        "groups": {kind: summarise([r for r in rows if r["kind"] == kind])
                   for kind in ("ordinary", "shot", "goal", "tv_goal")},
        "by_source": {source: {kind: summarise([r for r in rows if r["kind"] == kind and r["source"] == source])
                               for kind in ("ordinary", "shot", "goal")} for source in ("stadium", "tv")},
        "goals": [r for r in rows if r["kind"] in ("goal", "tv_goal")],
        "shots": [r for r in rows if r["kind"] == "shot"],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    for kind, group in report["groups"].items():
        print(kind, json.dumps(group))


if __name__ == "__main__":
    main()
