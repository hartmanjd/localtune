#!/usr/bin/env bash
# Sets up a rented GPU and starts the benchmark sweep: every step of the rental runbook in one go.
# Copy it to the rental together with the localtune wheel, then run it inside tmux:
#   bash ~/rental_setup.sh            full sweep with tuning (6 to 16 hours)
#   bash ~/rental_setup.sh --quick    each model and use once, usual settings (about 1 to 2 hours)
#   bash ~/rental_setup.sh --medium   the quick sweep, then about 70 runs with part of each model
#                                     in RAM (about 3 to 5 hours, about 100 runs in all)
# Safe to run again: steps that are already done are skipped or finish quickly.

set -euo pipefail

MODE="${1:-}"
TUNE="--tune"
SWEEP_TIME="6 to 16 hours"
if [ "$MODE" = "--quick" ] || [ "$MODE" = "--medium" ]; then
  TUNE=""
  SWEEP_TIME="about 1 to 2 hours"
fi

MODELS="llama3.1:8b qwen3:14b qwen3:14b-q8_0 gpt-oss:20b gemma4:26b qwen3:30b qwen3.8:27b"

# Layer counts from localtune's model table (src/localtune/data/models.csv), for the medium
# runs below. Dense models move whole layers to RAM (--gpu-layers); mixture-of-experts models
# keep every layer on the GPU and move only some layers' experts to RAM (--cpu-moe).
DENSE="llama3.1:8b:32 qwen3:14b:40 qwen3:14b-q8_0:40 qwen3.8:27b:65"
MOE="gpt-oss:20b:24 gemma4:26b:30 qwen3:30b:48"

step() { printf '\n\033[1;35m== %s ==\033[0m\n' "$1"; }

# The sweep takes hours and must survive a dropped SSH connection, so it has to run in tmux.
if [ -z "${TMUX:-}" ]; then
  echo "Not inside tmux. Start it first with: tmux   then run this script again."
  exit 1
fi

step "1/7 Install uv"
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Some rental images already come with uv; either way, uv's tools (localtune) land here.
export PATH="$HOME/.local/bin:$PATH"

step "2/7 Install localtune from the wheel"
WHEEL=$(ls -t "$HOME"/localtune-*.whl 2>/dev/null | head -n 1 || true)
if [ -z "$WHEEL" ]; then
  echo "No localtune-*.whl in $HOME. Copy it here with scp first."
  exit 1
fi
uv tool install --force "$WHEEL"

step "3/7 Install Ollama"
if ! command -v ollama >/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

step "4/7 Start Ollama"
if ! curl -s http://localhost:11434/api/version >/dev/null; then
  OLLAMA_CONTEXT_LENGTH=32768 nohup ollama serve > "$HOME/ollama.log" 2>&1 &
fi
for _ in $(seq 30); do
  curl -s http://localhost:11434/api/version >/dev/null && break
  sleep 1
done
ollama list

step "5/7 Download the 7 models (10 to 20 min the first time)"
for m in $MODELS; do
  ollama pull "$m"
done

step "6/7 Install llama.cpp and check the machine"
localtune doctor --install
localtune info

echo "To leave it running: press Ctrl+B, let go, press D."
echo "Before you stop or destroy this machine: copy ~/.localtune/results.jsonl home first!"

if [ "$MODE" != "--medium" ]; then
  step "7/7 Start the sweep ($SWEEP_TIME)"
  # shellcheck disable=SC2086
  localtune sweep $MODELS $TUNE
  exit 0
fi

# The medium sweep teaches the prediction model how this computer's CPU and RAM behave when a
# model doesn't all fit on the GPU, which the quick sweep on a big GPU never shows. Each model
# runs with a set share of it in RAM, with all the CPU threads this machine allows and with
# half of them. Every finished run leaves a marker in DONE, so running the script again carries
# on where it stopped instead of repeating runs.
DONE="$HOME/.localtune/medium-done"
mkdir -p "$DONE"

once() {
  local key="$1"
  shift
  if [ -e "$DONE/$key" ]; then
    return
  fi
  echo "-- localtune $*"
  if localtune "$@"; then
    touch "$DONE/$key"
  else
    echo "That run failed; carrying on with the next one."
    touch "$DONE/$key"
  fi
}

# The threads this machine allows: its CPUs, capped by the container's CPU limit if it has one.
# Newer containers (cgroup v2) keep the limit in cpu.max, older ones (v1) in two cpu.cfs files.
ALL=$(nproc)
QUOTA=max
if [ -r /sys/fs/cgroup/cpu.max ]; then
  read -r QUOTA PERIOD < /sys/fs/cgroup/cpu.max
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
  QUOTA=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
  PERIOD=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
fi
if [ "$QUOTA" != "max" ] && [ "$QUOTA" -gt 0 ] 2>/dev/null; then
  LIMIT=$(( (QUOTA + PERIOD - 1) / PERIOD ))
  [ "$LIMIT" -lt "$ALL" ] && ALL=$LIMIT
fi
HALF=$(( ALL > 1 ? ALL / 2 : 1 ))

step "7/8 Quick sweep: each model and use once ($SWEEP_TIME)"
if [ ! -e "$DONE/sweep" ]; then
  # shellcheck disable=SC2086
  localtune sweep $MODELS
  touch "$DONE/sweep"
fi

step "8/8 Runs with part of each model in RAM (about 2 to 3 hours, $ALL and $HALF threads)"
for entry in $DENSE; do
  model="${entry%:*}"
  layers="${entry##*:}"
  for pct in 90 75 50 25 10; do
    on_gpu=$(( layers * pct / 100 ))
    for t in "$ALL" "$HALF"; do
      once "$model-gpu$pct-t$t" run "$model" --use chat --gpu-layers "$on_gpu" --threads "$t"
    done
  done
  half=$(( layers / 2 ))
  once "$model-gpu50-code" run "$model" --use code --gpu-layers "$half" --threads "$ALL"
done
for entry in $MOE; do
  model="${entry%:*}"
  layers="${entry##*:}"
  for pct in 25 50 75 100; do
    in_ram=$(( layers * pct / 100 ))
    for t in "$ALL" "$HALF"; do
      once "$model-moe$pct-t$t" run "$model" --use chat --cpu-moe "$in_ram" --threads "$t"
    done
  done
  half=$(( layers / 2 ))
  once "$model-moe50-code" run "$model" --use code --cpu-moe "$half" --threads "$ALL"
done

echo "Medium sweep finished. Copy ~/.localtune/results.jsonl home before you destroy this machine."
