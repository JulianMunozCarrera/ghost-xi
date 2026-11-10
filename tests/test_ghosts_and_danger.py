import numpy as np

from ghostxi.analysis import analyse_shot, start_frame, ghost_defense, LOOK_BACK
from ghostxi.danger import danger, space, attack_control, ZONE
from ghostxi.ghost import make_ghosts, ghost_speed
from ghostxi.model_numpy import GhostModel

STILL = np.zeros((1, 2))


def test_ghosts_start_on_top_of_the_real_defenders(fake_clip):
    ghosts = make_ghosts(GhostModel(), fake_clip, start=40)
    assert np.allclose(ghosts[:41], fake_clip.defenders[:41])
    assert np.isfinite(ghosts[41:]).all()
    assert ghosts.shape == fake_clip.defenders.shape


def test_ghosts_stay_on_or_near_the_pitch(fake_clip):
    ghosts = make_ghosts(GhostModel(), fake_clip, start=20)
    assert np.abs(ghosts[..., 0]).max() < 60 and np.abs(ghosts[..., 1]).max() < 40


def test_solo_ghost_only_replaces_one_defender_at_a_time(fake_clip):
    model = GhostModel()
    solo = make_ghosts(model, fake_clip, start=20, solo=True, only_frames=[120])
    team = make_ghosts(model, fake_clip, start=20, solo=False, only_frames=[120])
    assert np.isfinite(solo[120]).all() and np.isfinite(team[120]).all()
    assert np.isnan(solo[119]).all()          # frames we did not ask for stay empty
    assert not np.allclose(solo[120], team[120])


def test_no_ghosts_beyond_ten_seconds(fake_clip):
    longer = fake_clip
    ghosts = make_ghosts(GhostModel(), longer, start=0)
    assert np.isfinite(ghosts[199]).all()     # 9.95 s after the start: fine
    short_fps = type(fake_clip)("x", "y", 10.0, fake_clip.ball, fake_clip.attackers, fake_clip.defenders, -1, 6, 180)
    ghosts = make_ghosts(GhostModel(), short_fps, start=0)
    assert np.isnan(ghosts[150]).all()        # at 10 fps frame 150 is 15 s later: too far


def test_ghost_speed_uses_half_a_second():
    ghosts = np.zeros((30, 2, 2))
    ghosts[:, 0, 0] = np.arange(30) * 0.2     # one ghost moves 0.2 m per frame
    speed = ghost_speed(ghosts, 20, fps=20.0)
    assert np.allclose(speed[0], [4.0, 0.0])  # 0.2 m per frame at 20 fps = 4 m/s
    assert np.allclose(speed[1], [0.0, 0.0])


def test_danger_is_high_when_the_attacker_is_alone_and_low_when_marked():
    attacker = np.array([[41.5, 0.0]])
    far_defender = np.array([[0.0, 0.0]])
    assert danger(attacker, STILL, far_defender, STILL) > 0.95
    assert danger(far_defender, STILL, attacker, STILL) < 0.05
    assert abs(danger(attacker, STILL, attacker, STILL) - 0.5) < 1e-6


def test_adding_a_defender_never_raises_the_danger():
    rng = np.random.default_rng(0)
    for _ in range(20):
        attackers = rng.uniform([25, -20], [50, 20], (4, 2))
        defenders = rng.uniform([30, -20], [52, 20], (4, 2))
        extra = np.vstack([defenders, rng.uniform([30, -20], [52, 20], (1, 2))])
        zeros4, zeros5 = np.zeros((4, 2)), np.zeros((5, 2))
        assert danger(attackers, zeros4, extra, zeros5) <= danger(attackers, zeros4, defenders, zeros4) + 1e-9


def test_control_is_a_chance_between_0_and_1():
    rng = np.random.default_rng(1)
    control = attack_control(ZONE, rng.uniform(20, 50, (5, 2)), rng.normal(0, 2, (5, 2)),
                             rng.uniform(20, 50, (5, 2)), rng.normal(0, 2, (5, 2)))
    assert control.min() >= 0 and control.max() <= 1 and len(control) == len(ZONE)


def test_space_ignores_the_goalkeeper():
    defenders = np.array([[10.0, 0.0], [1.0, 0.0]])
    assert space(np.zeros(2), defenders, keeper=1) == 10.0
    assert space(np.zeros(2), defenders, keeper=-1) == 1.0


def test_shot_analysis_has_sensible_numbers(fake_clip):
    result = analyse_shot(GhostModel(), fake_clip)
    assert start_frame(fake_clip) == 180 - int(LOOK_BACK * 20)   # frame 180 at 20 fps, minus the look-back
    assert result["seconds"] == LOOK_BACK
    assert result["compared"] == 7
    assert len(result["off_by"]) == 7 and min(result["off_by"]) >= 0
    assert 0 <= result["danger_real"] <= 1 and 0 <= result["danger_ghost"] <= 1


def test_defender_off_screen_at_the_start_gets_no_ghost(fake_tv_clip):
    model = GhostModel()
    for solo in (False, True):
        ghosts = make_ghosts(model, fake_tv_clip, start=80, solo=solo, only_frames=[140, 170])
        assert np.isnan(ghosts[140, 2]).all()            # defender 2 was off screen at frame 80
        assert np.isfinite(ghosts[140, [0, 1, 3, 4, 5, 6]]).all()
        # Defender 4 left the picture at frame 150. He still has a ghost: we know where he set off from.
        assert np.isfinite(ghosts[170, 4]).all()


def test_ghost_defense_only_swaps_defenders_we_can_compare(fake_tv_clip):
    model = GhostModel()
    speeds = fake_tv_clip.speeds()
    ghosts = make_ghosts(model, fake_tv_clip, start=80, only_frames=[160, 170])
    spots, moving, swapped = ghost_defense(fake_tv_clip, ghosts, 170, speeds)
    assert list(swapped) == [True, True, False, True, False, True, True]
    assert np.allclose(spots[2], fake_tv_clip.defenders[170, 2])     # no ghost: stays where he is
    assert np.isnan(spots[4]).all()                                  # off screen now: not part of either defense
    assert np.allclose(spots[0], ghosts[170, 0])


def test_shot_analysis_skips_defenders_it_cannot_compare(fake_tv_clip):
    result = analyse_shot(GhostModel(), fake_tv_clip)
    # At the shot (frame 180) defender 4 is off screen. At the start (frame 80) defender 2 was.
    assert result["compared"] == 5
    assert np.isnan(result["off_by"][2]) and np.isnan(result["off_by"][4])
    assert np.isfinite([result["off_by"][i] for i in (0, 1, 3, 5, 6)]).all()


def test_danger_ignores_players_who_are_off_screen():
    attackers = np.array([[41.5, 0.0], [np.nan, np.nan]])
    defenders = np.array([[0.0, 0.0], [np.nan, np.nan]])
    zeros = np.zeros((2, 2))
    assert danger(attackers, zeros, defenders, zeros) == danger(attackers[:1], zeros[:1], defenders[:1], zeros[:1])
    assert danger(np.full((1, 2), np.nan), STILL, defenders[:1], STILL) == 0.0   # no attacker in sight
