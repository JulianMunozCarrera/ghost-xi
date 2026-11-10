"""
Puts all the numbers in one place: results.json

  1. How close do ghosts move to real defenders? (from training, 3 way cross validation)
  2. What do the ghosts say about shots and goals compared with normal play?

Every match is judged by the model that never saw it.

    python scripts/evaluate.py --data data/metrica/data
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ghostxi.analysis import clip_around_shot, study
from ghostxi.control import attack_share, danger_zone
from ghostxi.data import load_all, shots, STEP
from ghostxi.model import load, ghost_paths
from ghostxi.windows import make_windows

METHODS = ["stand_still", "keep_running", "follow_ball", "ghost_model", "ghost_model_solo"]
COLUMNS = ["at_1s", "at_3s", "at_5s", "average_over_5s"]


def combine_scores(folder, names):
    """Average the three test scores, counting bigger test sets more."""
    scores = [json.load(open(os.path.join(folder, f"scores_{n}.json"))) for n in names]
    clips = np.array([s["test_clips"] for s in scores], dtype=float)
    table = {m: {c: round(float(sum(s[m][c] * k for s, k in zip(scores, clips)) / clips.sum()), 2) for c in COLUMNS}
             for m in METHODS}
    return {"test_clips": int(clips.sum()), "per_match": dict(zip(names, scores)), "combined": table}


def normal_play(match, net):
    """
    For comparison: ordinary 5 second clips that end with the ball in
    the defending third. Returns the same measurements as for shots,
    taken at the end of each clip.
    """
    clips = make_windows([match], every=5)
    near_goal = clips["ball"][:, -1, 0] < -17.5
    clips = {k: v[near_goal] for k, v in clips.items()}
    team = ghost_paths(net, clips)
    solo = ghost_paths(net, clips, solo=True)
    spots, weights = danger_zone()
    rows = []
    for i in range(len(team)):
        on, att_on = clips["defender_on"][i], clips["attacker_on"][i]
        outfield = on & ~clips["keeper"][i]
        real, attackers = clips["defenders"][i], clips["attackers"][i]
        real_speed = (real[-1] - real[-2]) / STEP
        ghost_speed = (team[i, -1] - team[i, -2]) / STEP
        attacker_speed = (attackers[-1] - attackers[-2]) / STEP
        gap = np.linalg.norm(solo[i, -1] - real[-1], axis=-1)[outfield]
        rows.append({
            "danger_real": attack_share(real[-1][on], real_speed[on], attackers[-1][att_on], attacker_speed[att_on], spots, weights),
            "danger_ghost": attack_share(team[i, -1][on], ghost_speed[on], attackers[-1][att_on], attacker_speed[att_on], spots, weights),
            "mean_gap": float(gap.mean()), "biggest_gap": float(gap.max()),
        })
    return rows


def shot_rows(match, net):
    rows = []
    for shot in shots(match):
        spell, shot_index = clip_around_shot(match, shot)
        if spell is None:
            continue
        result = study(spell, shot_index, net)
        outfield = result["defender_on"] & ~result["keeper"]
        gap = result["gap_at_shot"][outfield]
        step = result["shot_step"]
        rows.append({"goal": shot["goal"], "seconds_of_play": step * STEP,
                     "danger_real": float(result["danger_real"][step]), "danger_ghost": float(result["danger_ghost"][step]),
                     "mean_gap": float(gap.mean()), "biggest_gap": float(gap.max())})
    return rows


def summary(rows):
    extra = np.array([r["danger_real"] - r["danger_ghost"] for r in rows])
    return {
        "count": len(rows),
        "danger_real": round(float(np.mean([r["danger_real"] for r in rows])), 3),
        "danger_ghost": round(float(np.mean([r["danger_ghost"] for r in rows])), 3),
        "extra_danger_vs_ghost": round(float(extra.mean()), 3),
        "extra_danger_plus_or_minus": round(float(extra.std(ddof=1) / np.sqrt(len(extra))), 3),
        "share_where_real_was_more_dangerous": round(float((extra > 0).mean()), 3),
        "mean_gap_m": round(float(np.mean([r["mean_gap"] for r in rows])), 2),
        "biggest_gap_m": round(float(np.mean([r["biggest_gap"] for r in rows])), 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--models", default="models")
    parser.add_argument("--out", default="results.json")
    args = parser.parse_args()

    matches = load_all(args.data)
    report = {"ghost_accuracy": combine_scores(args.models, [m.name for m in matches])}

    all_shots, normal = [], []
    for match in matches:
        net = load(os.path.join(args.models, f"ghost_without_{match.name}.pt"))
        all_shots += shot_rows(match, net)
        normal += normal_play(match, net)
    # Only compare like with like: shots that had the full 5 seconds of open play before them.
    full = [r for r in all_shots if r["seconds_of_play"] >= 4.8]
    report["what_ghosts_found"] = {
        "goals": summary([r for r in all_shots if r["goal"]]),
        "all_shots": summary(all_shots),
        "shots_with_5s_of_play": summary(full),
        "normal_play_near_goal": summary(normal),
    }
    print(json.dumps({k: v for k, v in report["ghost_accuracy"].items() if k != "per_match"}, indent=1))
    print(json.dumps(report["what_ghosts_found"], indent=1))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
