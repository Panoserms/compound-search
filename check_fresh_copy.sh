#!/usr/bin/env bash
# Check that the project reproduces from nothing but the files that would be in a repository.
#
#   ./check_fresh_copy.sh /path/to/an/empty/or/new/folder
#
# It copies the project without the virtual environment, the downloaded data and the
# stored embeddings, installs the pinned packages into a new environment, downloads the
# data, runs ./reproduce.sh there, and compares every JSON result with this folder's results/.
# Takes about 45 minutes on an M2 with 8 GB (most of it encoding the passages).
#
# Note: reproduce.sh repeats the frozen test run. That is a repeat with unchanged systems,
# not a second attempt. Do not use this script to try out a changed system.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:?give a folder for the copy}"
mkdir -p "$TARGET"
if [ -n "$(ls -A "$TARGET")" ]; then echo "the folder $TARGET is not empty" >&2; exit 1; fi

echo "=== copy (as a repository would have it) ==="
rsync -a \
  --exclude '.venv' --exclude '__pycache__' --exclude '.pytest_cache' --exclude '*.egg-info' --exclude '.DS_Store' \
  --exclude 'data/raw' --exclude 'data/cache/*.npy' \
  "$HERE"/ "$TARGET"/
cd "$TARGET"

echo "=== environment with the pinned versions ==="
python3.11 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
.venv/bin/pip install -q -e . --no-deps

echo "=== data ==="
.venv/bin/python -I -m compound_search.download --raw data/raw

echo "=== reproduce.sh ==="
./reproduce.sh

echo "=== compare with the saved results ==="
.venv/bin/python -I -m compound_search.compare_results "$HERE/results" "$TARGET/results" --out "$TARGET/comparison.json"
echo "The copy gives the same results."
