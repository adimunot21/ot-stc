"""Phase 1 / R4 — inspect leader↔follower alignment for one episode.

Open question R4 (PROJECT_PLAN.md §9): are ``action`` (leader command Lᵉ) and
``observation.state`` (follower state Fᵉ) in the same calibrated units, and is
there a clean one-step command→state lag — i.e. should we align ``state[t+k]``
to ``action[t]`` before building the OT cost matrix?

This script answers it empirically on a single episode:
  1. Overlays action vs state per joint (visual: do they track, with a delay?).
  2. Sweeps integer lags k and reports the per-joint z-scored MSE between
     ``action[t]`` and ``state[t+k]``; the argmin lag is the candidate offset.
  3. Prints a recommendation and saves a figure to docs/findings/.

It runs against the public SO101 dataset by default so the R4 decision can be
explored before the local pilot recording exists. Re-run with your own pilot
repo id once recorded, then write the final decision into docs/findings/.

Run:
    conda activate ot-stc
    python scripts/01_inspect_alignment.py
    python scripts/01_inspect_alignment.py lerobot/svla_so101_pickplace 2
Expected: a printed lag table + recommendation, and a saved PNG under
docs/findings/. With the public data the best lag is typically small (0–1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures, never open a window
import matplotlib.pyplot as plt
import numpy as np

from ot_stc.io.dataset import EpisodeTrajectory, load_episode_trajectories

# --- Config (each value justified) ----------------------------------------- #
CONFIG = {
    "repo_id": "lerobot/svla_so101_pickplace",  # default public set; override via argv[1]
    "episode_index": 0,  # which episode to inspect; override via argv[2]
    "max_lag": 5,  # sweep k = 0..max_lag; >1 step lag on a 30 fps arm would be surprising
    "findings_dir": "docs/findings",  # small PNG artifact lives with the writeup
}


def zscore_per_joint(x: np.ndarray) -> np.ndarray:
    """Z-score each joint (column) independently so per-joint scale (R5) doesn't
    let the gripper dominate the aggregate alignment error."""
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)  # guard constant joints
    return (x - mean) / std


def lag_error(leader: np.ndarray, follower: np.ndarray, lag: int) -> float:
    """Mean squared error between ``leader[t]`` and ``follower[t+lag]`` over the
    overlapping window, after per-joint z-scoring. Lower = better alignment."""
    n = leader.shape[0]
    if lag >= n:
        return float("nan")
    lead = zscore_per_joint(leader)[: n - lag]
    foll = zscore_per_joint(follower)[lag:]
    return float(np.mean((lead - foll) ** 2))


def per_joint_best_lag(
    leader: np.ndarray, follower: np.ndarray, max_lag: int
) -> list[int]:
    """Argmin lag for each joint independently (scale-invariant within a joint)."""
    n, dof = leader.shape
    best = []
    for j in range(dof):
        errs = []
        for k in range(max_lag + 1):
            lead = leader[: n - k, j]
            foll = follower[k:, j]
            errs.append(np.mean((lead - foll) ** 2))
        best.append(int(np.argmin(errs)))
    return best


def plot_overlay(ep: EpisodeTrajectory, out_path: Path) -> None:
    """Save a 2×3 grid overlaying leader vs follower per joint over time."""
    dof = ep.dof
    ncols = 3
    nrows = int(np.ceil(dof / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    t = np.arange(ep.num_frames) / ep.fps  # seconds

    for j in range(dof):
        ax = axes[j]
        ax.plot(t, ep.leader[:, j], label="leader (action)", lw=1.2)
        ax.plot(t, ep.follower[:, j], label="follower (state)", lw=1.2, alpha=0.8)
        ax.set_title(ep.joint_names[j], fontsize=9)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)
    for j in range(dof, len(axes)):
        axes[j].set_visible(False)
    for ax in axes[-ncols:]:
        ax.set_xlabel("time (s)")

    fig.suptitle(
        f"R4 alignment — {ep.task!r}  (episode {ep.episode_index}, T={ep.num_frames})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv: list[str]) -> int:
    repo_id = argv[1] if len(argv) > 1 else CONFIG["repo_id"]
    episode_index = int(argv[2]) if len(argv) > 2 else CONFIG["episode_index"]
    max_lag = CONFIG["max_lag"]

    print(f"Loading episode {episode_index} of {repo_id} ...")
    episodes = load_episode_trajectories(repo_id, episodes=[episode_index])
    ep = episodes[0]
    leader, follower = ep.leader, ep.follower
    print(f"Loaded: T={ep.num_frames}, DOF={ep.dof}, fps={ep.fps}, task={ep.task!r}\n")

    # --- Units sanity: same names => same calibrated space. Report raw ranges. #
    print("Per-joint ranges (leader vs follower) — confirms shared units/scale:")
    print(f"  {'joint':<18} {'leader[min,max]':>22} {'follower[min,max]':>22}")
    for j, name in enumerate(ep.joint_names):
        lr = f"[{leader[:, j].min():.1f}, {leader[:, j].max():.1f}]"
        fr = f"[{follower[:, j].min():.1f}, {follower[:, j].max():.1f}]"
        print(f"  {name:<18} {lr:>22} {fr:>22}")
    print()

    # --- Lag sweep (aggregate, z-scored) ------------------------------------ #
    print("Aggregate z-scored alignment MSE vs lag k  (state[t+k] ~ action[t]):")
    print(f"  {'k':>3} {'MSE':>10}")
    errors = []
    for k in range(max_lag + 1):
        err = lag_error(leader, follower, k)
        errors.append(err)
        print(f"  {k:>3} {err:>10.4f}")
    best_lag = int(np.nanargmin(errors))
    improvement = (errors[0] - errors[best_lag]) / errors[0] * 100 if errors[0] else 0.0

    joint_lags = per_joint_best_lag(leader, follower, max_lag)
    print("\nPer-joint best lag:")
    for name, k in zip(ep.joint_names, joint_lags, strict=True):
        print(f"  {name:<18} k={k}")

    # --- Recommendation ----------------------------------------------------- #
    print("\n" + "=" * 60)
    print("R4 RECOMMENDATION")
    print("=" * 60)
    print(f"Best aggregate lag: k={best_lag} "
          f"(reduces z-scored MSE by {improvement:.1f}% vs k=0).")
    if best_lag == 0:
        print("→ No meaningful one-step lag. Use action[t] vs state[t] directly "
              "(no shift) in the OT cost matrix.")
    elif improvement < 5.0:
        print(f"→ A k={best_lag} shift exists but the gain is marginal (<5%). "
              "Prefer no shift for simplicity unless the per-joint lags agree.")
    else:
        print(f"→ Consider aligning state[t+{best_lag}] to action[t] before the OT "
              "cost matrix; confirm the per-joint lags are consistent first.")
    print("This is a single-episode probe — confirm on the pilot, then record the "
          "decision in docs/findings/.\n")

    # --- Save figure -------------------------------------------------------- #
    findings_dir = Path(CONFIG["findings_dir"])
    findings_dir.mkdir(parents=True, exist_ok=True)
    safe_repo = repo_id.replace("/", "__")
    out_path = findings_dir / f"r4_alignment_{safe_repo}_ep{episode_index}.png"
    plot_overlay(ep, out_path)
    print(f"Saved overlay figure → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
