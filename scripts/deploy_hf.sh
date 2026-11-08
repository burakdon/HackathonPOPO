#!/usr/bin/env bash
# Publish the NeuroCue demo to a Hugging Face Space (Docker SDK, free CPU tier).
#
#   one-time:  pip install -U huggingface_hub        (gives you the `hf` command)
#              hf auth login                          (paste a token with WRITE access)
#   deploy:    ./scripts/deploy_hf.sh [owner/space]   default: <your HF user>/neurocue
#
# What it does: snapshots the committed tree (git archive HEAD, so transcripts and anything else
# ignored by git never leave the machine), puts the Space card on top as README.md, and uploads
# it as one commit. The Space then builds the Dockerfile and serves the console at
# https://<owner>-<space>.hf.space. Re-run after every push you want live.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v hf >/dev/null 2>&1 || { echo "hf CLI not found: pip install -U huggingface_hub"; exit 1; }

if [ -n "${1:-}" ]; then
  SPACE="$1"
else
  USER_NAME="$(hf auth whoami --format json 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("name") or d.get("user") or d["username"])')" \
    || { echo "not logged in: run 'hf auth login' with a write token, then retry"; exit 1; }
  SPACE="$USER_NAME/neurocue"
fi

echo "Space: $SPACE"
hf repos create "$SPACE" --type space --space-sdk docker --public --exist-ok >/dev/null

BUNDLE="$(mktemp -d)"
trap 'rm -rf "$BUNDLE"' EXIT
git archive --format=tar HEAD | tar -x -C "$BUNDLE"
cp deploy/hf/README.md "$BUNDLE/README.md"
cp deploy/hf/.gitattributes "$BUNDLE/.gitattributes"
rm -rf "$BUNDLE/.github" "$BUNDLE"/results/demo/*.mp4 "$BUNDLE"/results/demo/*.webm

hf upload "$SPACE" "$BUNDLE" . --repo-type space --delete '*' \
  --commit-message "deploy $(git rev-parse --short HEAD): $(git log -1 --pretty=%s | cut -c1-60)"

echo
echo "Uploaded. The Space builds the image now (5-10 min the first time; later builds reuse layers)."
echo "  page:  https://huggingface.co/spaces/$SPACE          (Logs tab shows the build)"
echo "  app:   https://${SPACE/\//-}.hf.space"
