import numpy as np
import pytest

from ghostxi.samples import MatchSamples
from ghostxi.tokens import build_tokens, moves_to_positions, TOKENS, FEATURES, FIRST_ATTACKER, FIRST_DEFENDER, MOVE_SCALE


def test_hidden_defender_keeps_no_trace_of_where_he_is_now(fake_clip):
    speeds = fake_clip.speeds()
    now, then = fake_clip.moment(100, speeds), fake_clip.moment(20, speeds)
    hidden = [i == 3 for i in range(7)]
    tokens, present = build_tokens(now, then, 4.0, hidden)

    assert tokens.shape == (TOKENS, FEATURES)
    token = tokens[FIRST_DEFENDER + 3]
    assert token[4] == 1                      # marked as hidden
    assert np.all(token[5:9] == 0)            # where he is now: blank
    assert np.all(token[16:18] == 0)          # how far he is from the ball now: blank
    assert np.any(token[9:11] != 0)           # where he was then: still there
    # If we move him "now", the token must not change at all.
    now["def"] = now["def"].copy()
    now["def"][3] += 9.0
    again, _ = build_tokens(now, then, 4.0, hidden)
    assert np.array_equal(again[FIRST_DEFENDER + 3], token)


def test_empty_slots_and_missing_players_are_not_present(fake_clip):
    speeds = fake_clip.speeds()
    now, then = fake_clip.moment(100, speeds), fake_clip.moment(20, speeds)
    now["att"] = now["att"].copy()
    now["att"][2] = np.nan                    # one attacker is off screen
    tokens, present = build_tokens(now, then, 4.0, [True] * 7)
    assert present.sum() == 1 + 5 + 7         # ball + 5 attackers + 7 defenders
    assert not present[1 + 2]
    assert np.all(tokens[~present] == 0)


def test_player_who_walks_into_the_picture_gets_a_token_with_a_blank_then(fake_tv_clip):
    speeds = fake_tv_clip.speeds()
    now, then = fake_tv_clip.moment(140, speeds), fake_tv_clip.moment(80, speeds)
    tokens, present = build_tokens(now, then, 3.0, [False] * 7)

    late_attacker = tokens[FIRST_ATTACKER + 3]     # on screen now, not then
    assert present[FIRST_ATTACKER + 3]
    assert late_attacker[20] == 1                  # flag: no "then"
    assert np.all(late_attacker[9:13] == 0) and np.all(late_attacker[18:20] == 0)
    assert np.any(late_attacker[5:7] != 0)         # his "now" is there

    normal = tokens[FIRST_ATTACKER + 0]
    assert normal[20] == 0 and np.any(normal[9:11] != 0)


def test_defender_with_no_then_cannot_be_hidden(fake_tv_clip):
    speeds = fake_tv_clip.speeds()
    now, then = fake_tv_clip.moment(140, speeds), fake_tv_clip.moment(80, speeds)
    tokens, present = build_tokens(now, then, 3.0, [True] * 7)
    # Defender 2 was off screen "then". Asked to hide him, we have nothing to go on, so he gets no token.
    assert not present[FIRST_DEFENDER + 2]
    assert present[FIRST_DEFENDER + 1]

    # Defender 4 is off screen at frame 160 but was on screen then. He can still be hidden.
    later = fake_tv_clip.moment(160, speeds)
    tokens, present = build_tokens(later, then, 4.0, [i == 4 for i in range(7)])
    assert present[FIRST_DEFENDER + 4] and tokens[FIRST_DEFENDER + 4, 4] == 1
    # But as a visible teammate he has no token, because he is not on screen now.
    tokens, present = build_tokens(later, then, 4.0, [False] * 7)
    assert not present[FIRST_DEFENDER + 4]


def test_moves_are_added_to_the_starting_spot():
    then_def = np.array([[10.0, 5.0], [20.0, -5.0]])
    answer = np.zeros((TOKENS, 2))
    answer[FIRST_DEFENDER] = [0.5, -0.2]      # the model says: moved 5 m forward, 2 m down
    spots = moves_to_positions(answer, then_def)
    assert np.allclose(spots, [[15.0, 3.0], [20.0, -5.0]])


@pytest.mark.parametrize("which", ["fake_match", "fake_tv_match"])
def test_fast_batch_builder_matches_the_simple_one(which, request):
    match = request.getfixturevalue(which)
    samples = MatchSamples(match)
    rng = np.random.default_rng(0)
    checked_targets = 0
    for _ in range(60):
        pick = rng.integers(len(samples.times))
        now = int(samples.times[pick])
        back = int((1 + rng.integers(samples.longest[pick] // 5)) * 5)
        then = now - back
        hide = rng.integers(11)
        tokens, present, hidden, target = samples.build(
            [now], [back], lambda usable: np.eye(11, dtype=bool)[[hide]] & usable)

        attacking = int(match.possession[now])
        sign = match.direction[(1, attacking)]

        def moment(frame):
            out = {"ball": match.ball[frame] * sign, "ball_v": samples.ball_speed[frame] * sign}
            for key, team in (("att", attacking), ("def", 1 - attacking)):
                rank = np.isnan(samples.xy[now, team, :, 0]) * 2 + np.isnan(samples.xy[then, team, :, 0])
                order = np.argsort(rank, kind="stable")[:11]
                out[key] = samples.xy[frame, team][order] * sign
                out[key + "_v"] = samples.speed[frame, team][order] * sign
                out[key + "_keeper"] = int(np.where(order == 0)[0][0]) if 0 in order else -1
            return out

        m_now, m_then = moment(now), moment(then)
        # The fast builder only hides a defender it saw at both moments.
        can_hide = not np.isnan(m_now["def"][hide]).any() and not np.isnan(m_then["def"][hide]).any()
        simple, simple_present = build_tokens(m_now, m_then, back / 25, [i == hide and can_hide for i in range(11)])
        assert np.allclose(simple, tokens[0], atol=1e-6)
        assert np.array_equal(simple_present, present[0])
        assert hidden[0, FIRST_DEFENDER + hide] == can_hide
        if can_hide:
            # The target is how far the hidden defender really moved.
            truth = (m_now["def"][hide] - m_then["def"][hide]) / MOVE_SCALE
            assert np.allclose(target[0, FIRST_DEFENDER + hide], truth, atol=1e-5)
            checked_targets += 1
    assert checked_targets > 10


def test_then_and_now_never_straddle_a_stoppage(fake_match):
    samples = MatchSamples(fake_match)
    for now, longest in zip(samples.times, samples.longest):
        assert not np.isnan(fake_match.ball[now - longest:now + 1, 0]).any()


def test_training_batch_hides_only_defenders(fake_match):
    samples = MatchSamples(fake_match)
    tokens, present, hidden, target = samples.random_batch(64, np.random.default_rng(0))
    assert not hidden[:, :FIRST_DEFENDER].any()
    assert np.all(tokens[hidden][:, 5:9] == 0)
    assert np.all(tokens[hidden][:, 20] == 0)          # a hidden defender always has his "then"
    assert np.all(target[~hidden] == 0)                # only hidden defenders are scored


def test_tv_matches_never_use_positions_the_camera_did_not_see(fake_tv_match):
    samples = MatchSamples(fake_tv_match)
    assert not samples.full_view
    unseen = ~fake_tv_match.seen
    assert np.isnan(samples.xy[unseen]).all()
    tokens, present, hidden, target = samples.random_batch(128, np.random.default_rng(1))
    assert hidden.any()
    # Fewer tokens than a full match: some players are off screen.
    assert present.sum(axis=1).mean() < 22


def test_full_view_matches_sometimes_pretend_to_be_tv(fake_match):
    samples = MatchSamples(fake_match)
    assert samples.full_view
    tokens, present, hidden, target = samples.random_batch(256, np.random.default_rng(2))
    counts = present.sum(axis=1)
    assert counts.max() == 23            # some examples show everyone
    assert counts.min() < 20             # some have players cut off by the pretend camera
