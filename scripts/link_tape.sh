#!/usr/bin/env bash
# Symlink existing TAPE directory into LOGIN/data/TAPE
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAPE_SRC="${1:-/storage/qiaoyr/TAPE}"
DEST="$REPO_ROOT/data/TAPE"
mkdir -p "$REPO_ROOT/data"
if [[ -e "$DEST" ]]; then
  echo "Already exists: $DEST"
  exit 1
fi
ln -s "$TAPE_SRC" "$DEST"
echo "Linked $DEST -> $TAPE_SRC"
echo "Or export LOGIN_TAPE_ROOT=$TAPE_SRC"
