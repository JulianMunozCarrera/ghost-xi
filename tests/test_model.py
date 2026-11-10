import os

import numpy as np
import pytest

from ghostxi.model_numpy import GhostModel, DEFAULT_WEIGHTS
from ghostxi.tokens import TOKENS, FEATURES


def test_numpy_model_gives_the_same_answer_as_pytorch(tmp_path):
    torch = pytest.importorskip("torch")
    from ghostxi.model import GhostNet

    torch.manual_seed(0)
    net = GhostNet().eval()
    path = os.path.join(tmp_path, "weights.npz")
    np.savez(path, **{name: value.numpy() for name, value in net.state_dict().items()})

    rng = np.random.default_rng(0)
    tokens = rng.normal(size=(5, TOKENS, FEATURES)).astype(np.float32)
    present = rng.random((5, TOKENS)) > 0.2
    present[:, 0] = True
    with torch.no_grad():
        expected = net(torch.from_numpy(tokens), torch.from_numpy(present)).numpy()
    answer = GhostModel(path).run(tokens, present)
    assert np.abs(answer - expected)[present].max() < 1e-4


def test_empty_slots_do_not_change_the_answer():
    model = GhostModel(DEFAULT_WEIGHTS)
    rng = np.random.default_rng(1)
    tokens = rng.normal(size=(TOKENS, FEATURES)).astype(np.float32)
    present = np.ones(TOKENS, dtype=bool)
    present[[5, 9, 20]] = False
    first = model.run(tokens, present)
    tokens[[5, 9, 20]] = 99.0                 # put junk in the empty slots
    second = model.run(tokens, present)
    assert np.allclose(first[present], second[present], atol=1e-5)


def test_player_order_does_not_matter():
    model = GhostModel(DEFAULT_WEIGHTS)
    rng = np.random.default_rng(2)
    tokens = rng.normal(size=(TOKENS, FEATURES)).astype(np.float32)
    present = np.ones(TOKENS, dtype=bool)
    shuffle = rng.permutation(TOKENS)
    first = model.run(tokens, present)
    second = model.run(tokens[shuffle], present[shuffle])
    assert np.allclose(first[shuffle], second, atol=1e-4)
