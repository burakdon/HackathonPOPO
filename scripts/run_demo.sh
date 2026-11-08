#!/usr/bin/env bash
# Start the NeuroCue patient console locally. Installs uv if missing, syncs the environment, starts the
# server on port 8765 (override with NEUROCUE_PORT) and opens the page.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  echo "installing uv (Python package manager)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
export PATH="$HOME/.local/bin:$PATH"
PORT="${NEUROCUE_PORT:-8765}"
echo "syncing Python environment (first time takes a few minutes: torch, sentence-transformers)…"
"$UV" sync --quiet
echo
echo "Starting the console at http://localhost:$PORT  (first start builds nothing: caches are in the repo)"
echo "Stop with Ctrl-C."
( sleep 6; command -v open >/dev/null && open "http://localhost:$PORT" || true ) &
NEUROCUE_PORT="$PORT" "$UV" run python -m neurocue.demo.server
