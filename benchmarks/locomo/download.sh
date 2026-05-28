#!/usr/bin/env bash
set -euo pipefail
# Download the LoCoMo-10 dataset (snap-research/locomo).
DATA_DIR="$(dirname "$0")/data"
mkdir -p "$DATA_DIR"
DEST="$DATA_DIR/locomo10.json"
if [ -f "$DEST" ]; then
  echo "[skip] $DEST already present"
  exit 0
fi
URL="https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
echo "[fetch] $URL -> $DEST"
curl -fL "$URL" -o "$DEST"
echo "[ok] $(wc -c < "$DEST") bytes"
