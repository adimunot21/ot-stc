# OT-STC — Project Plan

**Optimal Transport Stratified Curation of Teleoperation Demonstrations for Low-Cost Robotic Arms**

Platform: LeRobot SO101 (leader + follower) · Policy: ACT · OT engine: POT

---

## 1. Goal statement & success criteria

### Goal
Use optimal transport (OT) over joint-space trajectories to (a) measure how faithfully the SO101 follower executed the human's leader command *within each demonstration*, (b) measure how *diverse* demonstrations are relative to one another, and (c) use both signals to curate a teleoperation dataset before training an ACT policy — then test whether geometry-aware curation produces a better or more stable policy than naive baselines.

### The key data insight
A LeRobot recording already contains both trajectories as separate columns of the same parquet file:
- `action` column = the **leader** command signal `Lᵉ` (what the human asked for)
- `observation.state` column = the **follower** actual joint state `Fᵉ` (what the hardware did)

Both are 6-D (6 joint angles). The leader–follower fidelity gap is therefore the OT distance between these two columns of one episode. **No custom dual-recording rig is needed** — standard `lerobot-record` captures everything.

### Success criteria ("done" looks like)
The project is **done** when the full pipeline runs reproducibly end to end:

1. A recorded 100-episode SO101 dataset of a precision pick-and-place task.
2. A per-episode intra-episode quality score (Sinkhorn OT cost between `action` and `observation.state`) for all 100 episodes.
3. An N×N inter-episode diversity matrix + Wasserstein barycenter + a manifold plot (UMAP) of demonstration space.
4. Curated 50% datasets under two criteria (quality-only and quality+diversity Pareto), built as valid `LeRobotDataset`s.
5. Five trained ACT conditions (A–E below), each over ≥5 seeds, on RunPod.
6. An evaluation table: success rate, std-dev, and robustness-to-perturbation per condition, with a paired statistical test across seeds.
7. Joint-phase fidelity diagnostic maps derived from the transport plans.
8. A written `docs/findings/` report.

**Honesty clause:** success is *not* defined as "OT curation beats random by ≥X%." That is the **hypothesis under test**. A clean, well-controlled *negative* result (curation ≈ random) plus the hardware-diagnostic contribution still satisfies "done." We are measuring an effect, not manufacturing one.

---

## 2. Target users / use case

| User | Why they care |
|---|---|
| **You (portfolio / research)** | A self-contained, novel-use-of-OT study on real low-cost hardware. Demonstrates ML + robotics + applied optimal transport end-to-end. |
| **SO101 / LeRobot community** | The joint-phase fidelity maps are a reusable diagnostic: "which joints and which task phases is *your* hardware losing accuracy on." Independent of whether curation helps. |
| **Anyone doing robot BC on a budget** | A drop-in, unsupervised, model-free dataset-curation pre-step (no influence functions, no trained proxy model required). |

---

## 3. Hardware constraints & how we work within them

| Resource | Spec | Constraint | Strategy |
|---|---|---|---|
| GPU | GTX 1650, **4 GB VRAM** | Too small to train ACT comfortably (ACT wants ≥8–12 GB at batch 8 + image encoders). | **Do not train locally.** Local GPU is for tiny smoke tests only (batch 1–2, a few steps). All real training (conditions A–E × ≥5 seeds) runs on **RunPod A40 (48 GB)**. |
| CPU | i7-9750H, 6c/12t | Fine. | OT is CPU-bound and embarrassingly parallel — all Sinkhorn/EMD/barycenter work runs here. The 100×100 pairwise matrix is <30 min on CPU. |
| RAM | 32 GB | Comfortable. | Whole 100-episode tabular dataset fits in memory many times over. |
| Disk | 1 TB SSD | Comfortable. | Dataset (~a few GB incl. video) + checkpoints fit easily. |
| Arms | Leader + follower SO101 | — | Standard teleoperation recording. |
| Cameras | Overhead + wrist | — | Both recorded; used for ACT image conditioning, not for OT (OT is joint-space only). |

**Division of labour:** Lenovo = data collection + all OT/curation/analysis. RunPod = training + (optionally) the heavier eval renders. The follower arm must be physically present to *evaluate* policies, so eval runs locally with the trained checkpoint pulled down from the Hub.

---

## 4. System architecture & data flow

```
┌──────────────────────── LENOVO (local) ─────────────────────────┐
│                                                                  │
│  SO101 leader ──teleop──► SO101 follower                         │
│        │                        │                                │
│        └──── lerobot-record ─────┘                               │
│                    │                                             │
│                    ▼                                             │
│        LeRobotDataset (parquet + mp4)                            │
│        per episode:  action[T,6]  ·  observation.state[T,6]      │
│                    │                                             │
│        ┌───────────┴────────────┐                               │
│        ▼                        ▼                                │
│   io/dataset.py           (cameras → ACT only)                  │
│   extract Lᵉ, Fᵉ                                                 │
│        │                                                         │
│        ▼                                                         │
│   ot/quality.py      Sinkhorn(Lᵉ, Fᵉ)  → QualityScore[e]         │
│   ot/diversity.py    pairwise Sinkhorn(Fⁱ,Fʲ) → D[N,N]          │
│                      free-support barycenter → F*                │
│   ot/transport_maps.py  per-joint / per-phase fidelity          │
│        │                                                         │
│        ▼                                                         │
│   curation/pareto.py  → keep-lists for conditions C, D          │
│        │                                                         │
│        ▼                                                         │
│   build curated LeRobotDatasets  → push to HF Hub                │
└──────────────────────────────┬───────────────────────────────────┘
                               │ HF Hub
                               ▼
┌──────────────────────── RUNPOD A40 ─────────────────────────────┐
│   lerobot-train (ACT) × {A,B,C,D,E} × seeds  → checkpoints → Hub │
└──────────────────────────────┬───────────────────────────────────┘
                               │ pull checkpoint
                               ▼
┌──────────────────────── LENOVO (local) ─────────────────────────┐
│   lerobot-record w/ policy (eval) on real follower              │
│   50 trials/condition + perturbation → results CSV              │
│   viz/ → manifold plot, fidelity heatmaps, results table        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Technology choices & justification

| Choice | Over what | Why |
|---|---|---|
| **POT (`ot`)** for OT | DTW; custom EMD | Gives a **transport plan** (the diagnostic payload), principled distribution-level distance, native multivariate + unequal-length support, and `sinkhorn2`/`emd2`/`free_support_barycenter` out of the box. DTW collapses multivariate poorly and yields no plan. |
| **Entropic Sinkhorn** (`sinkhorn2`) | Exact EMD everywhere | Fast and stable for the 100×100 pairwise sweep; smooth/regularised plans read better as fidelity maps. Keep exact `emd2` available for spot-checking individual episodes. |
| **Time-augmented support** (append normalised t as a 7th channel) | Plain unordered point clouds | Treating each timestep as an unordered sample throws away temporal order, so OT could match leader-step 5 to follower-step 50 if they're geometrically close. Augmenting each sample to `[joints(6), λ·t]` penalises cross-time matches and keeps the lag-sensitivity that *is the whole point*. λ is tunable; ablate it. |
| **Per-joint z-score before cost matrix** | Raw joint units | The gripper joint moves on a different scale/units than the shoulder joints; without standardisation it would dominate the squared-Euclidean cost. |
| **UMAP** (+ t-SNE cross-check) | t-SNE alone | UMAP preserves global cluster structure better, which is what "do my 100 demos form 2–3 strategies or one blob?" actually asks. Keep sklearn t-SNE as a sanity cross-check. |
| **ACT via LeRobot** | Diffusion Policy; custom BC | It's the community-standard baseline on SO101, reproducible via `lerobot-train --policy.type=act`, and keeps the comparison about *curation*, not architecture. |
| **conda env** | bare venv | `ffmpeg`/`torchcodec` video-decode deps install cleanly from conda-forge (pinned 7.1.1), which is the documented LeRobot path. |
| **RunPod A40** | local GTX 1650 | 4 GB VRAM can't hold ACT training; A40's 48 GB runs it cheaply (~$10–20 for all conditions). |

---

## 6. Phase-by-phase breakdown

> Phase 0 is always environment setup. Each phase names its deliverable and what is **testable** before moving on. We build smallest-verifiable-thing first.

### Phase 0 — Environment & hardware bring-up
- **Build:** conda env, install LeRobot+Feetech+analysis stack, find ports, find cameras, calibrate both arms, confirm teleoperation, install the `ot_stc` package editable.
- **Deliverable:** `scripts/00_check_setup.py` that imports every key library, prints versions, and confirms POT/UMAP/LeRobot load.
- **Testable:** `lerobot-teleoperate` moves the follower from the leader; `00_check_setup.py` exits 0; cameras enumerate.

### Phase 1 — Data layer + 5-episode pilot
- **Build:** `io/dataset.py` — load a `LeRobotDataset`, slice it into per-episode arrays, return `(Lᵉ, Fᵉ)` as `np.ndarray[T,6]` from `action` and `observation.state`. Record a **5-episode pilot** to exercise the whole record→load path.
- **Deliverable:** function `load_episode_trajectories(repo_id) -> list[(L,F)]`.
- **Testable:** `tests/test_dataset.py` on the pilot: 5 episodes, both arrays `[T,6]`, T matches metadata, no NaNs, action/state are different (gap is non-zero).

### Phase 2 — Intra-episode quality score
- **Build:** `ot/quality.py` — per-episode Sinkhorn cost between `Lᵉ` and `Fᵉ` with per-joint standardisation + optional time channel. Lower = better fidelity.
- **Deliverable:** `quality_score(L, F, cfg) -> float` and a batch runner `scripts/02_compute_quality.py` → `experiments/results/quality.csv`.
- **Testable:** `tests/test_quality.py` — `score(X, X) ≈ 0`; injecting a constant offset into `F` strictly increases the score (monotonicity); ordering is stable across `reg` values.

### Phase 3 — Full data collection (100 demos)
- **Build:** record 100 episodes of the precision task, varied object positions; push to Hub.
- **Deliverable:** the real `LeRobotDataset` (100 episodes) + `quality.csv` for all 100.
- **Testable:** dataset loads, `num_episodes == 100`, every episode gets a finite score; eyeball worst/best episodes against the score.

### Phase 4 — Inter-episode diversity & manifold
- **Build:** `ot/diversity.py` — pairwise Sinkhorn `D[N,N]` over follower trajectories; free-support Wasserstein barycenter `F*`; `viz/manifold.py` UMAP/t-SNE embedding coloured by quality.
- **Deliverable:** `D.npy`, barycenter, `manifold.png`.
- **Testable:** `D` is symmetric, zero diagonal, non-negative; embedding renders; visually check for clusters vs blob.

### Phase 5 — OT-stratified curation
- **Build:** `curation/pareto.py` — distance-to-barycenter + quality → Pareto front; produce keep-lists for **C (quality-only)** and **D (quality+diversity)**; materialise curated `LeRobotDataset`s; compute condition **E** per-episode loss weights `1/QualityScore`.
- **Deliverable:** keep-index JSONs, two curated datasets on the Hub, a weights file for E.
- **Testable:** `tests/test_curation.py` — each curated set has exactly 50 episodes; C and D are *different* sets; weights are finite and positive.

### Phase 6 — Training (RunPod)
- **Build:** `scripts/06_train_all.sh` driving `lerobot-train --policy.type=act` for A–E × seeds.
- **Deliverable:** ≥25 checkpoints (5 conditions × 5 seeds), pushed to Hub, with wandb loss curves.
- **Testable:** every run completes; losses converge; checkpoints load.

### Phase 7 — Evaluation
- **Build:** `scripts/07_evaluate.sh` — run each checkpoint on the real follower, 50 trials/condition, plus an object-position-perturbation sweep.
- **Deliverable:** `experiments/results/eval.csv` (success, std, robustness).
- **Testable:** harness logs every trial; per-condition aggregates computed; counts == 50.

### Phase 8 — Analysis & diagnostics
- **Build:** `ot/transport_maps.py` + `viz/diagnostics.py` — from intra-episode plans, build per-joint × per-phase fidelity heatmaps; paired stats (e.g. paired t-test / bootstrap CIs across seeds) on A–E.
- **Deliverable:** fidelity heatmaps, final results table, `docs/findings/` writeup.
- **Testable:** heatmaps render; statistical comparison reported with CIs (not just point estimates).

---

## 7. Major dependencies

| Library | Handles |
|---|---|
| `lerobot[feetech]` | Robot control, teleoperation, `lerobot-record`, `LeRobotDataset`, ACT training/eval. Pulls `torch`. |
| `POT` (`import ot`) | Optimal transport: `sinkhorn2`, `emd2`, `free_support_barycenter`, transport plans. |
| `umap-learn` | Manifold embedding of demonstration space. |
| `scikit-learn` | t-SNE cross-check, metrics, PCA, standardisation. |
| `numpy` / `scipy` | Core numerics, distance matrices. |
| `pandas` / `pyarrow` | Read parquet directly, build results tables/CSVs. |
| `matplotlib` / `seaborn` | Manifold plots, fidelity heatmaps. |
| `pyyaml` | `config/*.yaml` loading. |
| `python-dotenv` | Load `.env` (HF token, wandb). |
| `tqdm` | Progress bars for the pairwise sweep. |
| `wandb` *(optional, via lerobot)* | Training loss/metric logging. |
| `pytest` / `ruff` | Tests + lint/format. |

*(No Node/npm — this is a pure-Python project.)*

---

## 8. Directory structure

```
ot-stc/
├── config/
│   ├── robot.yaml            # ports, camera indices, record settings
│   └── experiment.yaml       # OT / curation / training / eval params
├── data/
│   └── raw/                  # pointer/notes; real data lives in HF cache (gitignored)
├── src/ot_stc/
│   ├── io/dataset.py         # LeRobotDataset → per-episode (L=action, F=state)
│   ├── ot/quality.py         # intra-episode Sinkhorn fidelity score
│   ├── ot/diversity.py       # pairwise distances + Wasserstein barycenter
│   ├── ot/transport_maps.py  # joint-phase fidelity diagnostics
│   ├── curation/pareto.py    # Pareto-front curation, build curated datasets
│   ├── viz/manifold.py       # UMAP / t-SNE
│   ├── viz/diagnostics.py    # fidelity heatmaps, results plots
│   └── utils/config.py       # load yaml + .env
├── scripts/
│   ├── 00_check_setup.py
│   ├── 01_record_demos.sh
│   ├── 02_compute_quality.py
│   ├── 03_compute_diversity.py
│   ├── 04_curate.py
│   ├── 05_build_curated_datasets.py
│   ├── 06_train_all.sh       # RunPod
│   └── 07_evaluate.sh
├── notebooks/
├── experiments/
│   ├── configs/              # per-condition train configs
│   └── results/              # quality.csv, eval.csv, summaries (small files tracked)
├── tests/
│   ├── test_dataset.py
│   ├── test_quality.py
│   └── test_curation.py
├── docs/
│   ├── PROJECT_PLAN.md
│   └── findings/
├── .env.example
├── .gitignore
├── environment.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 9. Known risks & open questions (resolve early)

| # | Risk / question | When it bites | Mitigation / resolution |
|---|---|---|---|
| R1 | **Time-order loss** in OT-over-unordered-samples. | Phase 2 | Time-augment support (`[joints, λt]`); ablate λ; sanity-check that a deliberately lagged synthetic episode scores worse. |
| R2 | **4 GB VRAM** can't train ACT. | Phase 6 | RunPod A40; local GPU only for batch-1 smoke tests. Confirmed in plan. |
| R3 | **Effect size within noise** (curation only ~2–3% over random). | Phase 7 | ≥5 seeds/condition; report paired test + bootstrap CIs; frame negative result as a valid outcome. |
| R4 | **`action` units vs `observation.state` units / one-step lag.** Are both in the same calibrated joint space? Is there a deterministic 1-step offset between command and state? | Phase 1 | Inspect `meta/info.json` + `meta/stats.json`; plot `action` vs `state` for one episode; decide whether to align `state[t+1]` to `action[t]`. Resolve **before** Phase 2. |
| R5 | **Per-joint scale** (gripper dominates cost). | Phase 2 | Per-joint z-score using dataset stats before building `C`. |
| R6 | **Variable episode length T.** | Phase 2/4 | OT handles unequal supports natively; use uniform marginals normalised to sum 1. No truncation needed. |
| R7 | **Dataset format / import path drift** (`lerobot.datasets` vs older `lerobot.common.datasets`; v2.1 vs v3.0). | Phase 1 | Pin the installed `lerobot` version; detect import path at runtime with a try/except; record the version in `00_check_setup.py`. |
| R8 | **Task difficulty mis-set** (too easy → every demo works → no signal; too hard → none work). | Phase 3 | Pick cube-in-small-bowl precision task; run a quick 10-demo pilot policy to confirm the task sits in the informative middle band before committing 100 demos. |
| R9 | **fps / timestamp jitter** from USB latency affecting trajectory sampling. | Phase 3 | Record at fixed fps; check `timestamp` deltas; drop/flag episodes with large gaps. |

---

## 10. Compute & time budget (from scope)

| Step | Cost |
|---|---|
| Collect 100 demos | ~3–5 h (≈2 min/episode incl. resets) |
| OT pairwise 100×100 | <30 min CPU |
| Curation | <1 min/condition |
| Training (5 conditions × ~3 h, A40) | ~15 A40-h ≈ $10–20 (×seeds scales this; budget accordingly) |
| Evaluation | 250 real-robot trials ≈ ~10 h |
