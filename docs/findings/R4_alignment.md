# R4 — leader↔follower units & command→state lag

**Status: PRELIMINARY** (probed on the public dataset; must be re-confirmed on the
local pilot before Phase 2 is finalised).

**Question (PROJECT_PLAN.md §9, R4):** Are `action` (leader command Lᵉ) and
`observation.state` (follower state Fᵉ) in the same calibrated units, and is there
a clean command→state lag — i.e. should we align `state[t+k]` to `action[t]`
before building the OT cost matrix?

**Probe:** `scripts/01_inspect_alignment.py` on `lerobot/svla_so101_pickplace`
episode 0 (lerobot 0.4.4), T=303, 30 fps. Figure:
`docs/findings/r4_alignment_lerobot__svla_so101_pickplace_ep0.png`.

## Findings

1. **Same units / shared space — confirmed.** `action` and `observation.state`
   carry identical joint names (`shoulder_pan.pos` … `gripper.pos`), all `.pos`,
   and per-joint min/max ranges line up closely (e.g. shoulder_pan leader
   `[1.7, 85.7]` vs follower `[1.9, 84.8]`). No unit conversion needed.

2. **Per-joint scale differs — confirmed (R5).** gripper ≈ `[0, 21]` vs
   shoulder_lift ≈ `[-100, 2]`. Per-joint z-score before any OT cost matrix is
   mandatory, as planned.

3. **The lag is ~4 steps, not one.** Sweeping `state[t+k] ~ action[t]` with
   per-joint z-scored MSE:

   | k | 0 | 1 | 2 | 3 | **4** | 5 |
   |---|---|---|---|---|-------|---|
   | MSE | 0.0626 | 0.0429 | 0.0276 | 0.0181 | **0.0156** | 0.0198 |

   Best lag **k=4** (~133 ms at 30 fps), and it is **identical across all six
   joints**. Aligning cuts the z-scored MSE by ~75% vs k=0. This contradicts the
   plan's "one-step lag" wording — on this hardware/recording the follower trails
   the command by roughly four control steps.

## Decision (PRELIMINARY)

The lag is **uniform across all six joints** (same k, ~133 ms). That uniformity is
the key: a per-joint mechanical fault would show *different* lags per joint. A
common offset on every joint is **common-mode pipeline latency** (teleop read →
command → actuation → state read), i.e. a property of the *recording rig*, not a
per-demo execution flaw. So:

- **Remove it as a single global constant before scoring.** Shift
  `state[t+k] ↔ action[t]` with one k applied to every episode and every joint,
  then the OT score measures genuine tracking *error*, not constant latency.
  Without this, the time-augmented cost (R1: append `λ·t`, which penalises
  cross-time matches) would read a perfectly-tracking-but-delayed follower as
  fidelity loss.
- **Never fit k per-episode.** A per-episode (or per-joint) best-fit k would
  absorb real lateness into the alignment and *launder out exactly the badness we
  want to measure* — a demo where the follower genuinely lagged the command would
  be silently re-aligned to look clean. One global k only.

### How the shift is applied (Phase 2)

1. Latency is stored in **milliseconds** (`ot.lag_ms` in `config/experiment.yaml`)
   so it survives an fps change. Steps are derived:
   `k = round(lag_ms * fps / 1000)` (133 ms @ 30 fps → k = 4).
2. Shift `follower[k:]` against `leader[:-k]` (state trails the command),
   then **truncate both to the overlap window** (length `T − k`).
3. Apply **per-joint z-scoring *after* alignment** (R5), so standardisation uses
   the aligned, truncated window — not the raw, misaligned series.

The loader (`io/dataset.py`) still returns **raw, unaligned** arrays by design; the
shift lives downstream in the scorer so the choice stays explicit and auditable.

## Diagnostic note (separate from the curation score)

The uniform latency is itself a **reportable hardware-profile result**: "this
SO101 teleop rig trails the leader by ~133 ms (≈4 steps @ 30 fps), consistently
across all joints." It is reported *independently* of the per-episode quality
score (which is computed *after* the latency is removed). The two answer different
questions — rig latency characterises the setup; the OT score characterises each
demonstration's execution fidelity given that setup.

## To do on the pilot (blocks Phase 2)

- [ ] Re-run `scripts/01_inspect_alignment.py <pilot_repo> <ep>` on ≥3 pilot
      episodes; confirm k is **stable across episodes** and still uniform across
      joints (expect ~constant; the absolute value is hardware-dependent and may
      differ from the public dataset's k=4).
- [ ] Set `ot.lag_ms` in `experiment.yaml` to the confirmed pilot latency.
- [ ] Update this file's status to **FINAL** with the chosen `lag_ms` and rationale.
