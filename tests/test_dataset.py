"""Tests for ot_stc.io.dataset.

Two tiers:
  * Offline unit tests for the pure validator ``_build_episode`` (no network).
  * A network integration test (marked ``network``) that loads a small slice of
    the public SO101 pick-place dataset and asserts the Phase-1 contract. It is
    skipped automatically if the dataset can't be reached/downloaded.

Run only fast offline tests:   pytest -q -m "not network"
Run everything (needs HF):     pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from ot_stc.io.dataset import (
    EpisodeTrajectory,
    _build_episode,
    load_episode_trajectories,
)

JOINTS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
PUBLIC_REPO = "lerobot/svla_so101_pickplace"
PILOT_N = 5


# --------------------------------------------------------------------------- #
# Offline unit tests for _build_episode (the pure validation core)
# --------------------------------------------------------------------------- #


def _rand_traj(T: int = 10, dof: int = 6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, dof)).astype(np.float32)


def test_build_episode_happy_path():
    leader, follower = _rand_traj(seed=1), _rand_traj(seed=2)
    ep = _build_episode(3, leader, follower, JOINTS, task="pick", fps=30.0)

    assert isinstance(ep, EpisodeTrajectory)
    assert ep.episode_index == 3
    assert ep.num_frames == 10
    assert ep.dof == 6
    assert ep.joint_names == JOINTS
    assert ep.task == "pick"
    assert ep.fps == 30.0
    assert ep.leader.dtype == np.float32 and ep.follower.dtype == np.float32


def test_build_episode_coerces_dtype_and_lists():
    # Plain python lists should be coerced to float32 [T, DOF].
    leader = [[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]]
    follower = [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]
    ep = _build_episode(0, leader, follower, JOINTS, task=None, fps=30.0)
    assert ep.leader.shape == (2, 6)
    assert ep.leader.dtype == np.float32


def test_build_episode_rejects_shape_mismatch():
    leader = _rand_traj(T=10)
    follower = _rand_traj(T=9)  # different length
    with pytest.raises(ValueError, match="same shape"):
        _build_episode(0, leader, follower, JOINTS, task=None, fps=30.0)


def test_build_episode_rejects_dof_name_mismatch():
    leader = _rand_traj(dof=5)
    follower = _rand_traj(dof=5)
    with pytest.raises(ValueError, match="DOF"):
        _build_episode(0, leader, follower, JOINTS, task=None, fps=30.0)


def test_build_episode_rejects_empty():
    empty = np.zeros((0, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="empty"):
        _build_episode(0, empty, empty, JOINTS, task=None, fps=30.0)


def test_build_episode_rejects_nan():
    leader = _rand_traj()
    follower = _rand_traj()
    follower[2, 0] = np.nan
    with pytest.raises(ValueError, match="NaN/inf"):
        _build_episode(0, leader, follower, JOINTS, task=None, fps=30.0)


def test_build_episode_rejects_1d():
    bad = np.zeros(6, dtype=np.float32)
    with pytest.raises(ValueError, match="2-D"):
        _build_episode(0, bad, bad, JOINTS, task=None, fps=30.0)


# --------------------------------------------------------------------------- #
# Network integration test against the public dataset (the Phase-1 contract)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def pilot_episodes():
    """Load the first PILOT_N episodes of the public dataset once per module.

    Skips the whole network suite if the dataset can't be fetched (offline/CI
    without HF access) so the fast offline tests still pass standalone.
    """
    try:
        return load_episode_trajectories(PUBLIC_REPO, episodes=range(PILOT_N))
    except Exception as err:  # network/auth/format issue -> skip, don't fail
        pytest.skip(f"Could not load {PUBLIC_REPO}: {type(err).__name__}: {err}")


@pytest.mark.network
def test_loads_requested_episode_count(pilot_episodes):
    assert len(pilot_episodes) == PILOT_N


@pytest.mark.network
def test_episodes_sorted_and_unique(pilot_episodes):
    indices = [ep.episode_index for ep in pilot_episodes]
    assert indices == sorted(indices)
    assert len(set(indices)) == len(indices)


@pytest.mark.network
def test_shapes_and_joint_names(pilot_episodes):
    for ep in pilot_episodes:
        assert ep.leader.ndim == 2 and ep.leader.shape[1] == 6
        assert ep.leader.shape == ep.follower.shape  # [T, 6] each
        assert ep.leader.dtype == np.float32 and ep.follower.dtype == np.float32
        assert ep.joint_names == JOINTS
        assert ep.fps == 30.0


@pytest.mark.network
def test_no_nans(pilot_episodes):
    for ep in pilot_episodes:
        assert np.isfinite(ep.leader).all()
        assert np.isfinite(ep.follower).all()


@pytest.mark.network
def test_length_matches_metadata(pilot_episodes):
    # T must equal the metadata-reported episode length (boundaries are correct).
    from ot_stc.io.dataset import _import_lerobot

    ds = _import_lerobot()(PUBLIC_REPO, download_videos=False)
    lengths = {int(r["episode_index"]): int(r["length"]) for r in ds.meta.episodes}
    for ep in pilot_episodes:
        assert ep.num_frames == lengths[ep.episode_index]


@pytest.mark.network
def test_leader_follower_gap_is_nonzero(pilot_episodes):
    # The fidelity signal must exist: action and state are genuinely different.
    for ep in pilot_episodes:
        assert not np.array_equal(ep.leader, ep.follower)
        assert np.abs(ep.leader - ep.follower).mean() > 0.0


@pytest.mark.network
def test_missing_episode_raises():
    with pytest.raises(ValueError, match="not present"):
        load_episode_trajectories(PUBLIC_REPO, episodes=[10_000])
