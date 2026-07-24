#!/usr/bin/env bash
# Download the three evaluation benchmarks used in the paper (§4).
# Requires the Hugging Face CLI:  pip install huggingface_hub
#
# Usage:  bash scripts/download_benchmarks.sh [TARGET_DIR]      (default: ./benchmarks)
set -euo pipefail
TARGET=${1:-benchmarks}
mkdir -p "$TARGET"

echo "==> IntentBench  (PhilipC/IntentBench)"
huggingface-cli download PhilipC/IntentBench  --repo-type dataset --local-dir "$TARGET/IntentBench"

echo "==> Daily-Omni   (liarliar/Daily-Omni)"
huggingface-cli download liarliar/Daily-Omni  --repo-type dataset --local-dir "$TARGET/Daily-Omni"

echo "==> WorldSense   (honglyhly/WorldSense)"
huggingface-cli download honglyhly/WorldSense --repo-type dataset --local-dir "$TARGET/WorldSense"

cat <<EOF

Done. Datasets are under $TARGET/.
Some of these ship videos as archives (.tar/.zip) — extract them first.
Then set the paths in your .env to the QA files and video roots, e.g.:

  IB_QA=$TARGET/IntentBench/qa.json
  IB_VIDEOS=$TARGET/IntentBench/videos
  DAILY_QA=$TARGET/Daily-Omni/qa.json
  DAILY_VIDEOS=$TARGET/Daily-Omni
  WORLD_QA=$TARGET/WorldSense/worldsense_qa.json
  WORLD_VIDEOS=$TARGET/WorldSense

Exact filenames/layout are set by each dataset card — check them and adjust the
paths above to match what was downloaded.
EOF
