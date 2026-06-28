#!/usr/bin/env bash
#
# Phase 1 — pilot/full teleoperation recording wrapper for the SO101.
#
# Reads ALL robot/recording parameters from config/robot.yaml (single source of
# truth — nothing hardcoded here) and the dataset target from .env, then builds
# and runs `lerobot-record`. Run a 5-episode pilot first, then re-run unchanged
# for the full set:
#
#   conda activate ot-stc
#   ./scripts/01_record_demos.sh --num-episodes 5     # pilot
#   ./scripts/01_record_demos.sh                      # full (uses config count)
#   ./scripts/01_record_demos.sh --no-push            # keep dataset local only
#
# Requires the `ot-stc` conda env to be active (provides lerobot-record + pyyaml).
set -euo pipefail

# --- Resolve repo-relative paths ------------------------------------------- #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROBOT_YAML="$REPO_ROOT/config/robot.yaml"
ENV_FILE="$REPO_ROOT/.env"

fail() { echo "ERROR: $*" >&2; exit 1; }

usage() {
    cat <<'USAGE'
Usage: scripts/01_record_demos.sh [--num-episodes N] [--no-push] [-h|--help]

  --num-episodes N   Override record.num_episodes from config (e.g. 5 for a pilot).
  --no-push          Record locally only; do not push the dataset to the HF Hub.
  -h, --help         Show this help.

All other parameters (ports, cameras, fps, task, episode/reset time) come from
config/robot.yaml. The dataset repo id is resolved from .env.
USAGE
}

# --- Parse arguments -------------------------------------------------------- #
NUM_EPISODES_OVERRIDE=""
PUSH_TO_HUB=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-episodes) NUM_EPISODES_OVERRIDE="${2:?--num-episodes requires a value}"; shift 2 ;;
        --num-episodes=*) NUM_EPISODES_OVERRIDE="${1#*=}"; shift ;;
        --no-push) PUSH_TO_HUB=false; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

# --- Preconditions ---------------------------------------------------------- #
command -v lerobot-record >/dev/null 2>&1 \
    || fail "lerobot-record not found on PATH. Activate the env: conda activate ot-stc"
[[ -f "$ROBOT_YAML" ]] || fail "config not found: $ROBOT_YAML"

# --- Parse robot.yaml (fps is the single source of truth) ------------------- #
# Emits shell-safe NAME=value lines (one per needed key); fails loudly on any
# missing/empty key so we never silently record with a bad parameter.
parse_config() {
    python3 - "$ROBOT_YAML" <<'PY'
import sys, shlex
try:
    import yaml
except ImportError:
    sys.stderr.write("pyyaml not found. Activate the env: conda activate ot-stc\n")
    sys.exit(4)

cfg = yaml.safe_load(open(sys.argv[1]))

def need(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur or cur[k] in (None, ""):
            sys.stderr.write("missing/empty config key: " + ".".join(keys) + "\n")
            sys.exit(3)
        cur = cur[k]
    return cur

def emit(name, val):
    print(f"{name}={shlex.quote(str(val))}")

emit("FOLLOWER_TYPE",  need(cfg, "follower", "type"))
emit("FOLLOWER_PORT",  need(cfg, "follower", "port"))
emit("FOLLOWER_ID",    need(cfg, "follower", "id"))
emit("LEADER_TYPE",    need(cfg, "leader", "type"))
emit("LEADER_PORT",    need(cfg, "leader", "port"))
emit("LEADER_ID",      need(cfg, "leader", "id"))
emit("FPS",            need(cfg, "record", "fps"))
emit("TASK",           need(cfg, "record", "task"))
emit("NUM_EPISODES",   need(cfg, "record", "num_episodes"))
emit("EPISODE_TIME_S", need(cfg, "record", "episode_time_s"))
emit("RESET_TIME_S",   need(cfg, "record", "reset_time_s"))

# Build the draccus camera Dict string lerobot-record expects, e.g.:
#   { overhead: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, ... }
cams = need(cfg, "cameras")
if not isinstance(cams, dict) or not cams:
    sys.stderr.write("config key 'cameras' must be a non-empty mapping\n"); sys.exit(3)
parts = []
for cam_name, c in cams.items():
    for field in ("type", "index_or_path", "width", "height", "fps"):
        if not isinstance(c, dict) or field not in c:
            sys.stderr.write(f"camera '{cam_name}' missing field '{field}'\n"); sys.exit(3)
    parts.append(
        f"{cam_name}: {{type: {c['type']}, index_or_path: {c['index_or_path']}, "
        f"width: {c['width']}, height: {c['height']}, fps: {c['fps']}}}"
    )
emit("CAMERAS", "{ " + ", ".join(parts) + "}")
PY
}

CONFIG_VALUES="$(parse_config)" || fail "could not parse $ROBOT_YAML (see message above)"
eval "$CONFIG_VALUES"

# --- Load .env and resolve the dataset repo id ------------------------------ #
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi
HF_USER="${HF_USER:-}"
DATASET_REPO_ID="${DATASET_REPO_ID:-}"

is_unset_or_placeholder() { [[ -z "$1" || "$1" == *your-hf-username* ]]; }

if is_unset_or_placeholder "$DATASET_REPO_ID"; then
    fail "DATASET_REPO_ID is missing or still a placeholder in $ENV_FILE.
       Set it to e.g.  DATASET_REPO_ID=<your-hf-user>/ot-stc-cube-bowl"
fi
if [[ "$DATASET_REPO_ID" == */* ]]; then
    REPO_ID="$DATASET_REPO_ID"                       # already namespaced
else
    is_unset_or_placeholder "$HF_USER" \
        && fail "DATASET_REPO_ID '$DATASET_REPO_ID' has no namespace and HF_USER is unset/placeholder in $ENV_FILE."
    REPO_ID="$HF_USER/$DATASET_REPO_ID"              # prepend the user namespace
fi

# --- Resolve episode count (override wins over config) ---------------------- #
NUM_EPISODES="${NUM_EPISODES_OVERRIDE:-$NUM_EPISODES}"
[[ "$NUM_EPISODES" =~ ^[1-9][0-9]*$ ]] \
    || fail "num-episodes must be a positive integer, got '$NUM_EPISODES'"

# --- Reminder always prints on exit (success, failure, or Ctrl-C) ----------- #
print_next_steps() {
    echo
    echo "============================================================"
    echo "R4 — DO THIS BEFORE PHASE 2:"
    echo "  Inspect leader->follower alignment on >=3 pilot episodes:"
    echo "    python scripts/01_inspect_alignment.py $REPO_ID 0"
    echo "    python scripts/01_inspect_alignment.py $REPO_ID 1"
    echo "    python scripts/01_inspect_alignment.py $REPO_ID 2"
    echo "  Confirm the lag k is STABLE across episodes (expect ~constant),"
    echo "  then set ot.lag_ms in config/experiment.yaml and mark"
    echo "  docs/findings/R4_alignment.md FINAL."
    echo "============================================================"
}
trap print_next_steps EXIT

# --- Build the lerobot-record command -------------------------------------- #
CMD=(
    lerobot-record
    --robot.type="$FOLLOWER_TYPE"
    --robot.port="$FOLLOWER_PORT"
    --robot.id="$FOLLOWER_ID"
    --robot.cameras="$CAMERAS"
    --teleop.type="$LEADER_TYPE"
    --teleop.port="$LEADER_PORT"
    --teleop.id="$LEADER_ID"
    --dataset.repo_id="$REPO_ID"
    --dataset.single_task="$TASK"
    --dataset.fps="$FPS"
    --dataset.num_episodes="$NUM_EPISODES"
    --dataset.episode_time_s="$EPISODE_TIME_S"
    --dataset.reset_time_s="$RESET_TIME_S"
    --dataset.push_to_hub="$PUSH_TO_HUB"
)

# --- Report resolved settings, then echo the exact command ------------------ #
echo "============================================================"
echo "OT-STC recording — resolved configuration"
echo "============================================================"
printf '  %-18s %s\n' "Dataset repo id:" "$REPO_ID"   # <- copy this into the inspect command
printf '  %-18s %s\n' "Task:"            "$TASK"
printf '  %-18s %s\n' "Episodes:"        "$NUM_EPISODES"
printf '  %-18s %s\n' "fps:"             "$FPS"
printf '  %-18s %s\n' "Episode time s:"  "$EPISODE_TIME_S"
printf '  %-18s %s\n' "Reset time s:"    "$RESET_TIME_S"
printf '  %-18s %s\n' "Follower:"        "$FOLLOWER_TYPE @ $FOLLOWER_PORT (id=$FOLLOWER_ID)"
printf '  %-18s %s\n' "Leader:"          "$LEADER_TYPE @ $LEADER_PORT (id=$LEADER_ID)"
printf '  %-18s %s\n' "Cameras:"         "$CAMERAS"
printf '  %-18s %s\n' "Push to Hub:"     "$PUSH_TO_HUB"
echo
echo "Running command:"
for arg in "${CMD[@]}"; do printf '    %q\n' "$arg"; done
echo

# Run as a child (not exec) so the EXIT trap fires and prints the next steps
# whether recording finishes, errors, or is interrupted with Ctrl-C.
"${CMD[@]}"
