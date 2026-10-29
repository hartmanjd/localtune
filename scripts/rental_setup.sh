#!/usr/bin/env bash
# Sets up a rented GPU and starts the benchmark sweep: every step of the rental runbook in one go.
# Copy it to the rental together with the localtune wheel, then run it inside tmux:
#   bash ~/rental_setup.sh
# Safe to run again: steps that are already done are skipped or finish quickly.

set -euo pipefail

MODELS="llama3.1:8b qwen3:14b qwen3:14b-q8_0 gpt-oss:20b gemma4:26b qwen3:30b qwen3.8:27b"

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
# shellcheck disable=SC1091
source "$HOME/.local/bin/env"

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

step "7/7 Start the sweep (6 to 16 hours)"
echo "To leave it running: press Ctrl+B, let go, press D."
echo "Before you stop or destroy this machine: copy ~/.localtune/results.jsonl home first!"
# shellcheck disable=SC2086
localtune sweep $MODELS --tune
