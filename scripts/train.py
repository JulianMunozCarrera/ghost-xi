"""
Trains the ghost model and measures how good it is.

Data: 3 matches tracked in the stadium (Metrica) and 20 matches tracked
from the TV broadcast (SkillCorner).

To test fairly we never score the model on a match it trained on. The
23 matches are split into 4 groups. We train 4 times, each time leaving
one group out, and test on the group that was left out. So every match
gets tested by a model that never saw it.

At the end we train one more model on all 23 matches. That is the one
the website uses, on goals from a third data source.

    python scripts/train.py --metrica data/metrica/data --skillcorner data/skillcorner/data
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ghostxi import data_metrica, data_skillcorner
from ghostxi.model import GhostNet
from ghostxi.samples import MatchSamples
from ghostxi.tokens import FIRST_DEFENDER, MOVE_SCALE, NEAR_SCALE, POSITION_SCALE

BATCH = 256
GROUPS = 4
SECONDS_AHEAD = (1, 2, 3, 5, 8)
DEFENDERS = slice(FIRST_DEFENDER, None)


def split_into_groups(matches):
    """
    Deal the matches into 4 groups, like dealing cards. The 3 stadium
    matches go into different groups, so every group has both kinds.
    Returns a list with the group number of each match.
    """
    groups, tv_dealt = [], 0
    for i, match in enumerate(matches):
        if match.source == "metrica":
            groups.append(i % GROUPS)
        else:
            groups.append((GROUPS - 1 - tv_dealt) % GROUPS)
            tv_dealt += 1
    return groups


def train(samples, steps, seed, label):
    """Train one model on a list of MatchSamples. Returns the trained network."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = GhostNet()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    # The learning rate starts high and fades out, so the last steps only make fine adjustments.
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=steps, pct_start=0.05)

    # Bigger matches get picked more often, so every moment has the same chance.
    sizes = np.array([len(s.times) for s in samples], dtype=np.float64)
    chance = sizes / sizes.sum()

    began, recent = time.time(), []
    for step in range(1, steps + 1):
        # One batch = 256 moments, taken from 4 randomly chosen matches.
        parts = [samples[i].random_batch(BATCH // 4, rng) for i in rng.choice(len(samples), 4, p=chance)]
        tokens, present, hidden, target = (torch.from_numpy(np.concatenate(x)) for x in zip(*parts))

        guess = net(tokens, present)
        # How many meters off is each guess? Only hidden defenders count.
        miss = (guess - target).norm(dim=-1) * MOVE_SCALE
        loss = miss[hidden].mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()
        schedule.step()

        recent.append(loss.item())
        if step % 1000 == 0 or step == steps:
            print(f"  [{label}] step {step}/{steps}  average miss {np.mean(recent):.2f} m  ({time.time() - began:.0f}s)", flush=True)
            recent = []
    return net.eval()


def follow_ball_fractions(samples, rng):
    """
    For the "move with the ball" rule: if the ball moves 10 m, how far
    does a defender usually move the same way? We measure that fraction
    on training data, once for length and once for width.
    """
    top, bottom = np.zeros(2), np.zeros(2)
    for s in samples:
        tokens, present, hidden, target = s.random_batch(2000, rng)
        ball_move = np.broadcast_to(tokens[:, :1, 14:16] * NEAR_SCALE, target.shape)[hidden]
        player_move = target[hidden] * MOVE_SCALE
        top += (ball_move * player_move).sum(axis=0)
        bottom += (ball_move ** 2).sum(axis=0)
    return top / bottom


def score(net, samples, fractions, seed=0):
    """
    Test on matches the model has not seen.

    For 1, 2, 3, 5 and 8 seconds ahead we hide defenders and compare the
    model's guess with where they really went. Two versions:
      one_hidden: hide a single defender (he can see his teammates)
      all_hidden: hide the whole defense
    Returns sums, so results from several matches can be added up.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for seconds in SECONDS_AHEAD:
        for mode in ("one_hidden", "all_hidden"):
            total = {"model": 0.0, "stand_still": 0.0, "follow_ball": 0.0, "copy_teammates": 0.0, "defenders_checked": 0}
            for s in samples:
                back = int(round(seconds * s.fps))
                now = s.times[s.longest >= back][::3]
                if len(now) == 0:
                    continue

                def choose(usable):
                    if mode == "all_hidden":
                        return usable
                    # Pick one usable defender per moment at random.
                    order = rng.random(usable.shape) * usable
                    pick = np.zeros_like(usable)
                    pick[np.arange(len(usable)), order.argmax(axis=1)] = True
                    return pick & usable

                tokens, present, hidden, target = s.build(now, np.full(len(now), back), choose)
                with torch.no_grad():
                    guess = np.concatenate([net(torch.from_numpy(tokens[i:i + 2048]), torch.from_numpy(present[i:i + 2048])).numpy()
                                            for i in range(0, len(tokens), 2048)])
                real = target * MOVE_SCALE
                miss = lambda g: float(np.linalg.norm(g - real, axis=-1)[hidden].sum())

                total["model"] += miss(guess * MOVE_SCALE)
                total["stand_still"] += miss(np.zeros_like(real))
                ball_move = tokens[:, :1, 14:16] * NEAR_SCALE
                total["follow_ball"] += miss(np.broadcast_to(ball_move * fractions, real.shape))

                # "Move like your teammates": the average move of the defenders we can see.
                defenders = tokens[:, DEFENDERS]
                shown = present[:, DEFENDERS] & ~hidden[:, DEFENDERS] & (defenders[..., 20] == 0)
                moves = (defenders[..., 5:7] - defenders[..., 9:11]) * POSITION_SCALE * shown[..., None]
                typical = moves.sum(axis=1) / np.maximum(shown.sum(axis=1), 1)[:, None]
                total["copy_teammates"] += miss(np.broadcast_to(typical[:, None, :], real.shape))
                total["defenders_checked"] += int(hidden.sum())
            out[(seconds, mode)] = total
    return out


def add_up(parts):
    """Add the sums from several test runs and turn them into averages (meters)."""
    table = {}
    for seconds in SECONDS_AHEAD:
        table[str(seconds)] = {}
        for mode in ("one_hidden", "all_hidden"):
            rows = [p[(seconds, mode)] for p in parts]
            count = sum(r["defenders_checked"] for r in rows)
            names = ["model", "stand_still", "follow_ball"] + (["copy_teammates"] if mode == "one_hidden" else [])
            table[str(seconds)][mode] = {n: round(sum(r[n] for r in rows) / max(count, 1), 3) for n in names}
            table[str(seconds)][mode]["defenders_checked"] = count
    return table


def save_weights(net, path):
    np.savez(path, **{name: value.numpy() for name, value in net.state_dict().items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrica", required=True, help="the 'data' folder of the Metrica repo")
    parser.add_argument("--skillcorner", required=True, help="the 'data' folder of the SkillCorner repo")
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--models", default="models")
    parser.add_argument("--out", default="results/model_results.json")
    parser.add_argument("--only", default=None, help="comma list of what to run, from: 1,2,3,4,final")
    args = parser.parse_args()
    only = set(args.only.split(",")) if args.only else {"1", "2", "3", "4", "final"}

    matches = data_metrica.load_all(args.metrica)
    print(f"loaded {len(matches)} Metrica matches", flush=True)
    matches += data_skillcorner.load_all(args.skillcorner, args.cache)
    print(f"loaded {len(matches)} matches in total", flush=True)
    samples = [MatchSamples(m) for m in matches]
    groups = split_into_groups(matches)
    moments = int(sum(len(s.times) for s in samples))
    minutes = sum((m.possession >= 0).sum() / m.fps / 60 for m in matches)
    print(f"{len(matches)} matches, {minutes:.0f} minutes of live play, {moments} moments to learn from", flush=True)

    os.makedirs(args.models, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for group in range(GROUPS):
        if str(group + 1) not in only:
            continue
        train_set = [s for s, g in zip(samples, groups) if g != group]
        test_set = [s for s, g in zip(samples, groups) if g == group]
        print(f"Round {group + 1}: testing on {len(test_set)} matches the model will not see", flush=True)
        net = train(train_set, args.steps, seed=group, label=f"round {group + 1}")
        save_weights(net, os.path.join(args.models, f"ghost_round{group + 1}.npz"))
        fractions = follow_ball_fractions(train_set, np.random.default_rng(100 + group))
        # Keep the raw sums per kind of tracking, so they can be added up across rounds later.
        raw = {}
        for kind, wanted in (("stadium_tracking", True), ("tv_tracking", False)):
            part = score(net, [s for s in test_set if s.full_view == wanted], fractions)
            raw[kind] = {f"{sec}|{mode}": v for (sec, mode), v in part.items()}
            print(f"  {kind}: 5 s ahead, all hidden", {k: round(v / max(part[(5, 'all_hidden')]['defenders_checked'], 1), 2)
                                                      for k, v in part[(5, "all_hidden")].items() if k in ("model", "stand_still", "follow_ball")}, flush=True)
        with open(os.path.join(args.models, f"round{group + 1}_scores.json"), "w") as f:
            json.dump(raw, f)

    if "final" in only:
        print("Final model: training on all matches", flush=True)
        net = train(samples, args.steps, seed=99, label="final")
        save_weights(net, os.path.join(args.models, "ghost.npz"))

    # Put the report together from whatever rounds have been run so far.
    done = [g for g in range(GROUPS) if os.path.exists(os.path.join(args.models, f"round{g + 1}_scores.json"))]
    if done:
        raws = [json.load(open(os.path.join(args.models, f"round{g + 1}_scores.json"))) for g in done]

        def unpack(raw_kind):
            return {(int(k.split("|")[0]), k.split("|")[1]): v for k, v in raw_kind.items()}

        report = {
            "matches": len(matches), "stadium_matches": sum(s.full_view for s in samples),
            "tv_matches": sum(not s.full_view for s in samples),
            "minutes_of_live_play": round(float(minutes)), "moments": moments,
            "steps": args.steps, "batch": BATCH, "rounds_done": len(done),
            "parameters": int(sum(p.numel() for p in GhostNet().parameters())),
            "tested_in_round": {m.name: g + 1 for m, g in zip(matches, groups)},
            "average": add_up([unpack(r[k]) for r in raws for k in r]),
            "by_source": {kind: add_up([unpack(r[kind]) for r in raws]) for kind in ("stadium_tracking", "tv_tracking")},
        }
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps({"5s_all_hidden": report["average"]["5"]["all_hidden"], "5s_one_hidden": report["average"]["5"]["one_hidden"]}))


if __name__ == "__main__":
    main()
