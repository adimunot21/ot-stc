# CLAUDE.md — OT-STC

Standing brief for Claude Code. Read `docs/PROJECT_PLAN.md` for the full plan; this file is the short, load-bearing version.

## What this project is
OT-STC uses **optimal transport** to (1) score how faithfully the SO101 follower executed the human's leader command within each teleoperation demo, (2) measure diversity across demos, and (3) curate the dataset before training an **ACT** policy via LeRobot — then test whether curation beats naive baselines.

## The one data fact that matters
A LeRobot recording stores both arms as two columns of the same parquet:
- `action` = **leader** command (Lᵉ), shape [T,6]
- `observation.state` = **follower** actual joint state (Fᵉ), shape [T,6]

The fidelity signal is the OT distance between these two columns per episode. No custom dual-recording rig exists or is needed.

## Hardware split (do not violate)
- **Local (this machine, GTX 1650 4 GB):** data collection, ALL OT/curation/analysis (CPU), policy evaluation on the real arm. NEVER train ACT locally beyond a batch-1 smoke test — 4 GB VRAM can't hold it.
- **RunPod A40:** all real ACT training (conditions A–E × ≥5 seeds).

## Environment
- conda env: `ot-stc` (python 3.10). Activate before anything: `conda activate ot-stc`.
- Package is `src/ot_stc/`, installed editable (`pip install -e ".[dev]"`). Import as `from ot_stc...`.
- LeRobot installed separately as `lerobot[feetech]`; let it own its torch version — never pin torch by hand.
- Config in `config/robot.yaml` and `config/experiment.yaml`. Secrets in `.env` (gitignored).

## Current state
- Phase 0 (env + scaffold) is **done**: env builds, imports verified, git pushed.
- **Next: Phase 1** — `src/ot_stc/io/dataset.py` (load LeRobotDataset → per-episode (L, F) arrays) + a 5-episode pilot recording. Resolve open question R4 on the pilot BEFORE Phase 2.

## Open question to resolve on the pilot (R4)
Confirm whether `action` and `observation.state` are in the same calibrated units and whether there's a clean one-step command→state lag (i.e. align `state[t+1]` to `action[t]`?). This decides the OT cost matrix. Plot one episode; decide; write the decision into `docs/findings/`.

## How to work here (build discipline)
- Write **complete, runnable** code. No placeholders, no TODO stubs. If a file needs 200 lines, write 200.
- Build smallest-testable-thing first. Every new module gets a matching `tests/test_*.py` before moving on.
- Give exact terminal commands for installing/running. Say what output to expect before running.
- After each working milestone: run `ruff check . && pytest -q`, then commit + push with a clear message.
- Match the import path at runtime defensively: try `from lerobot.datasets.lerobot_dataset import LeRobotDataset`, fall back to the older `lerobot.common.datasets...` path. Record the installed lerobot version.

## Conventions
- Per-joint z-score before building any OT cost matrix (gripper joint scale differs).
- Time-augment trajectories with a normalised time channel for intra-episode OT (preserves lag sensitivity); λ is a tunable in `experiment.yaml`.
- Results CSVs go in `experiments/results/` (tracked); heavy artifacts (parquet, mp4, checkpoints) stay gitignored.
- Use `numItermax` and `reg` from `config/experiment.yaml`; don't hardcode OT params.
