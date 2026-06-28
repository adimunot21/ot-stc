"""Data layer for OT-STC (Phase 1).

Load a ``LeRobotDataset`` and slice it into per-episode trajectories, returning
the **leader command** (``action``) and the **follower state**
(``observation.state``) as plain ``np.ndarray`` of shape ``[T, DOF]``.

The one data fact this module rests on (see PROJECT_PLAN.md §1):
    A LeRobot recording stores both arms as two columns of the same parquet:
      - ``action``            = leader command  Lᵉ   [T, 6]
      - ``observation.state`` = follower actual Fᵉ   [T, 6]
    The intra-episode fidelity signal is the OT distance between these columns.

Data contract (verified against ``lerobot/svla_so101_pickplace``, lerobot 0.4.4):
    - ``action`` and ``observation.state`` are both float32, shape (6,), with
      identical joint names: shoulder_pan/lift, elbow_flex, wrist_flex/roll,
      gripper — all in ``.pos`` units (same calibrated joint space).
    - Per-episode boundaries come from ``meta.episodes`` columns
      ``dataset_from_index`` / ``dataset_to_index`` (global positional indices
      into ``hf_dataset``) and ``length``.
    - Gripper range (~0–33) differs from shoulder (~-93–88): per-joint z-score is
      required before any OT cost matrix (handled in Phase 2, not here).

R4 (open until the pilot): whether to align ``state[t+1]`` to ``action[t]`` for a
one-step command→state lag is decided on the pilot plot and written to
``docs/findings/``. This loader deliberately returns **raw, unaligned** arrays;
any lag shift is applied downstream so the decision stays explicit and auditable.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

import numpy as np

# Column names are fixed by the LeRobot recording format.
LEADER_KEY = "action"
FOLLOWER_KEY = "observation.state"


def lerobot_version() -> str:
    """Return the installed lerobot version (or 'unknown' if unavailable)."""
    try:
        return version("lerobot")
    except PackageNotFoundError:  # pragma: no cover - lerobot always installed here
        return "unknown"


def _import_lerobot():
    """Import LeRobotDataset defensively across lerobot versions (R7).

    lerobot >= 0.4 lives at ``lerobot.datasets.lerobot_dataset``; older releases
    used ``lerobot.common.datasets.lerobot_dataset``. Try new first, fall back.
    """
    last_err: Exception | None = None
    for module_path in (
        "lerobot.datasets.lerobot_dataset",
        "lerobot.common.datasets.lerobot_dataset",
    ):
        try:
            module = importlib.import_module(module_path)
            return module.LeRobotDataset
        except ImportError as err:  # module path absent in this version
            last_err = err
    raise ImportError(
        "Could not import LeRobotDataset from any known path. "
        "Is `lerobot` installed in this env?"
    ) from last_err


@dataclass(frozen=True)
class EpisodeTrajectory:
    """One teleoperation episode as leader/follower joint trajectories.

    Attributes
    ----------
    episode_index : int
        The episode's index within its source dataset.
    leader : np.ndarray, shape [T, DOF], float32
        Leader command trajectory (the ``action`` column), Lᵉ.
    follower : np.ndarray, shape [T, DOF], float32
        Follower actual joint state (the ``observation.state`` column), Fᵉ.
    joint_names : tuple[str, ...]
        DOF joint names in column order (e.g. ``shoulder_pan.pos`` ...).
    task : str | None
        The natural-language task string for this episode, if recorded.
    fps : float
        Recording frame rate; lets downstream code convert frames<->seconds.
    """

    episode_index: int
    leader: np.ndarray
    follower: np.ndarray
    joint_names: tuple[str, ...]
    task: str | None
    fps: float

    @property
    def num_frames(self) -> int:
        """Episode length T in frames."""
        return int(self.leader.shape[0])

    @property
    def dof(self) -> int:
        """Number of joints (degrees of freedom)."""
        return int(self.leader.shape[1])


def _build_episode(
    episode_index: int,
    leader: np.ndarray,
    follower: np.ndarray,
    joint_names: Iterable[str],
    task: str | None,
    fps: float,
) -> EpisodeTrajectory:
    """Validate and assemble one ``EpisodeTrajectory`` (pure; no IO).

    Raises
    ------
    ValueError
        If shapes disagree, the DOF doesn't match ``joint_names``, the episode is
        empty, or any value is non-finite (NaN/inf).
    """
    leader = np.asarray(leader, dtype=np.float32)
    follower = np.asarray(follower, dtype=np.float32)
    names = tuple(joint_names)

    if leader.ndim != 2 or follower.ndim != 2:
        raise ValueError(
            f"Episode {episode_index}: expected 2-D [T, DOF] arrays, got "
            f"leader.ndim={leader.ndim}, follower.ndim={follower.ndim}."
        )
    if leader.shape != follower.shape:
        raise ValueError(
            f"Episode {episode_index}: leader {leader.shape} and follower "
            f"{follower.shape} must share the same shape."
        )
    if leader.shape[0] == 0:
        raise ValueError(f"Episode {episode_index}: empty episode (T == 0).")
    if leader.shape[1] != len(names):
        raise ValueError(
            f"Episode {episode_index}: DOF {leader.shape[1]} != number of joint "
            f"names {len(names)} ({names})."
        )
    if not (np.isfinite(leader).all() and np.isfinite(follower).all()):
        raise ValueError(f"Episode {episode_index}: contains NaN/inf values.")

    return EpisodeTrajectory(
        episode_index=int(episode_index),
        leader=leader,
        follower=follower,
        joint_names=names,
        task=task,
        fps=float(fps),
    )


def _episode_rows(meta, episodes: Iterable[int] | None):
    """Yield (episode_index, from_idx, to_idx, task) from dataset metadata.

    Boundaries come straight from ``meta.episodes`` (a HF table), so we never
    scan the per-frame ``episode_index`` column.
    """
    wanted = None if episodes is None else set(int(e) for e in episodes)
    for row in meta.episodes:
        ep_idx = int(row["episode_index"])
        if wanted is not None and ep_idx not in wanted:
            continue
        tasks = row.get("tasks")
        if isinstance(tasks, (list, tuple)):
            task = str(tasks[0]) if tasks else None
        else:
            task = None if tasks is None else str(tasks)
        yield ep_idx, int(row["dataset_from_index"]), int(row["dataset_to_index"]), task


def load_episode_trajectories(
    repo_id: str,
    *,
    episodes: Iterable[int] | None = None,
    root: str | None = None,
    revision: str | None = None,
    download_videos: bool = False,
) -> list[EpisodeTrajectory]:
    """Load per-episode (leader, follower) joint trajectories from a LeRobot dataset.

    Parameters
    ----------
    repo_id : str
        HF Hub repo id (e.g. ``"lerobot/svla_so101_pickplace"``) or a local
        dataset name resolvable via ``root``.
    episodes : iterable of int, optional
        Subset of episode indices to load. ``None`` (default) loads all episodes.
    root : str, optional
        Local dataset root; passed through to ``LeRobotDataset``.
    revision : str, optional
        Dataset revision/branch/tag for reproducibility.
    download_videos : bool, default False
        Whether to fetch the camera mp4s. OT is joint-space only, so we default
        to False — only parquet + metadata are needed, which keeps loading fast.

    Returns
    -------
    list[EpisodeTrajectory]
        One entry per requested episode, ordered by ``episode_index``. Each holds
        raw, unaligned ``leader`` and ``follower`` arrays of shape ``[T, DOF]``.

    Raises
    ------
    ValueError
        If any required column is missing, an episode is malformed, or an
        explicitly requested episode index is absent from the dataset.
    """
    lerobot_dataset_cls = _import_lerobot()
    dataset = lerobot_dataset_cls(
        repo_id,
        root=root,
        revision=revision,
        download_videos=download_videos,
    )
    meta = dataset.meta
    hf_dataset = dataset.hf_dataset

    for required in (LEADER_KEY, FOLLOWER_KEY):
        if required not in hf_dataset.column_names:
            raise ValueError(
                f"Dataset '{repo_id}' has no '{required}' column; found "
                f"{hf_dataset.column_names}."
            )

    joint_names = tuple(meta.features[LEADER_KEY]["names"])
    fps = float(meta.fps)

    rows = list(_episode_rows(meta, episodes))
    if episodes is not None:
        found = {ep for ep, *_ in rows}
        missing = sorted(set(int(e) for e in episodes) - found)
        if missing:
            raise ValueError(
                f"Requested episodes not present in '{repo_id}': {missing} "
                f"(dataset has {meta.total_episodes} episodes)."
            )

    trajectories: list[EpisodeTrajectory] = []
    for ep_idx, from_idx, to_idx, task in rows:
        chunk = hf_dataset.select(range(from_idx, to_idx))
        leader = np.asarray(chunk[LEADER_KEY], dtype=np.float32)
        follower = np.asarray(chunk[FOLLOWER_KEY], dtype=np.float32)
        trajectories.append(
            _build_episode(ep_idx, leader, follower, joint_names, task, fps)
        )

    trajectories.sort(key=lambda ep: ep.episode_index)
    return trajectories


if __name__ == "__main__":
    # Standalone sanity check against the public SO101 pick-place dataset.
    # Loads only the first 5 episodes (the pilot-sized slice) with videos off.
    import time

    REPO_ID = "lerobot/svla_so101_pickplace"
    PILOT_EPISODES = range(5)

    print(f"lerobot version: {lerobot_version()}")
    print(f"Loading {REPO_ID} episodes {list(PILOT_EPISODES)} (videos off)...")
    start = time.perf_counter()
    episodes = load_episode_trajectories(REPO_ID, episodes=PILOT_EPISODES)
    elapsed = time.perf_counter() - start
    print(f"Loaded {len(episodes)} episodes in {elapsed:.1f}s\n")

    print(f"joints ({episodes[0].dof}): {episodes[0].joint_names}")
    print(f"task: {episodes[0].task!r}  fps: {episodes[0].fps}\n")
    print(f"{'ep':>3} {'T':>5} {'mean|L-F|':>10} {'max|L-F|':>9}")
    for ep in episodes:
        gap = np.abs(ep.leader - ep.follower)
        print(f"{ep.episode_index:>3} {ep.num_frames:>5} {gap.mean():>10.4f} {gap.max():>9.4f}")

    # Cheap invariants the pilot test will assert formally.
    assert all(ep.leader.shape == ep.follower.shape for ep in episodes)
    assert all(ep.dof == len(ep.joint_names) for ep in episodes)
    assert all(np.isfinite(ep.leader).all() and np.isfinite(ep.follower).all() for ep in episodes)
    print("\nAll sanity checks passed.")
