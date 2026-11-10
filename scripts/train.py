"""
Trains the ghost model.

We have three matches. To test fairly, we never test on a match the
model trained on. So we train three times:

    train on matches 2 + 3, test on match 1
    train on matches 1 + 3, test on match 2
    train on matches 1 + 2, test on match 3

Every goal in the demo is analysed by the model that never saw that match.

    python scripts/train.py --data data/metrica/data
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ghostxi.data import load_all
from ghostxi.model import GhostNet, rollout, ghost_paths
from ghostxi.windows import make_windows, flip_sides
from ghostxi import baselines


def train_one(train_clips, epochs, seed=0, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    both = {k: np.concatenate([v, flip_sides(train_clips)[k]]) for k, v in train_clips.items()}
    data = {k: torch.as_tensor(v) for k, v in both.items()}
    count = len(data["ball"])

    net = GhostNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(epochs):
        net.train()
        # Start with short rollouts and make them longer as training goes on.
        # Learning 1 second first, then 5, is much easier than 5 straight away.
        steps = min(25, 5 + 20 * epoch // max(1, epochs // 2))
        total, batches = 0.0, 0
        order = torch.randperm(count)
        for i in range(0, count, 256):
            pick = order[i:i + 256]
            cut = slice(0, steps + 2)
            defenders = data["defenders"][pick][:, cut]
            # Half the time ghosts play together, half the time each ghost
            # plays next to the real team mates. We use both later.
            solo = bool(torch.rand(1) < 0.5)
            path = rollout(net, defenders, data["attackers"][pick][:, cut], data["ball"][pick][:, cut],
                           data["defender_on"][pick], data["attacker_on"][pick], data["keeper"][pick], solo=solo)
            miss = (path - defenders[:, 1:]).norm(dim=-1)                    # metres off, per player per step
            on = data["defender_on"][pick][:, None, :].expand_as(miss)
            loss = (miss * on).sum() / on.sum()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            total, batches = total + loss.item(), batches + 1
        schedule.step()
        log(f"  epoch {epoch + 1:2d}  rollout {steps * 0.2:.1f} s  average miss {total / batches:.2f} m")
    return net


def score(net, train_clips, test_clips):
    """Compare the model with the simple guesses on clips it never saw."""
    fractions = baselines.fit_follow_ball(train_clips)
    methods = {
        "stand_still": baselines.stand_still(test_clips),
        "keep_running": baselines.keep_running(test_clips),
        "follow_ball": baselines.follow_ball(test_clips, fractions),
        "ghost_model": ghost_paths(net, test_clips),
        "ghost_model_solo": ghost_paths(net, test_clips, solo=True),
    }
    table = {}
    for name, paths in methods.items():
        table[name] = {
            "average_over_5s": round(baselines.average_miss(paths, test_clips), 3),
            "at_1s": round(baselines.average_miss(paths, test_clips, 1), 3),
            "at_3s": round(baselines.average_miss(paths, test_clips, 3), 3),
            "at_5s": round(baselines.average_miss(paths[:, -1:], _last(test_clips)), 3),
        }
    return table


def _last(clips):
    """Clips cut down to the final frame only, for the 'at 5 s' score."""
    cut = dict(clips)
    cut["defenders"] = clips["defenders"][:, [0, -1]]
    return cut


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="folder with the Metrica sample games")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--out", default="models")
    parser.add_argument("--only-fold", type=int, default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    matches = load_all(args.data)
    results = {}
    for held_out, test_match in enumerate(matches):
        if args.only_fold is not None and held_out != args.only_fold:
            continue
        train_matches = [m for m in matches if m is not test_match]
        train_clips = make_windows(train_matches, every=2)
        test_clips = make_windows([test_match], every=5)
        print(f"Testing on {test_match.name}: {len(train_clips['ball'])} training clips, {len(test_clips['ball'])} test clips")
        began = time.time()
        net = train_one(train_clips, args.epochs)
        torch.save(net.state_dict(), os.path.join(args.out, f"ghost_without_{test_match.name}.pt"))
        results[test_match.name] = score(net, train_clips, test_clips)
        results[test_match.name]["test_clips"] = len(test_clips["ball"])
        print(json.dumps(results[test_match.name], indent=1), f"({time.time() - began:.0f} s)")
        with open(os.path.join(args.out, f"scores_{test_match.name}.json"), "w") as f:
            json.dump(results[test_match.name], f, indent=1)


if __name__ == "__main__":
    main()
