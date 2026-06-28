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

## Implication for the OT cost matrix

The time-augmented OT cost (R1: append `λ·t`) penalises cross-time matches, so a
raw ~4-step offset between Lᵉ and Fᵉ would register as *fidelity loss* even when
the follower is tracking perfectly, just delayed. Two options for Phase 2:

- **(a) Align then score:** shift `state[t+k]` ↔ `action[t]` (k from the pilot)
  so the score measures tracking *error*, not constant latency.
- **(b) Score raw:** treat latency as part of "fidelity" the policy should learn
  from. Simpler, but conflates latency with error.

**Leaning toward (a)** given the lag is large and joint-consistent. The loader
(`io/dataset.py`) returns **raw, unaligned** arrays by design; any shift is applied
downstream so the choice stays explicit. Make `lag` a tunable in
`config/experiment.yaml` and ablate it.

## To do on the pilot (blocks Phase 2)

- [ ] Re-run `scripts/01_inspect_alignment.py <pilot_repo> <ep>` on ≥3 pilot
      episodes; confirm k is stable (expect ~constant, hardware-dependent).
- [ ] Decide (a) vs (b); set `ot.lag` in `experiment.yaml`.
- [ ] Update this file's status to FINAL with the chosen k and rationale.
