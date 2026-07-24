#!/usr/bin/env bash
# Reproduce the §4 training set: HumanOmniV2's data with reasoning traces stripped.
#
# The committed data/matched_alldata.jsonl already lists the exact 30,217 training
# examples (questions + answers, traces removed) with repo-relative video paths.
# You only need this script to (a) get the videos and (b) regenerate the JSONL with
# absolute paths pointing at where you put them.
#
# Requires: git, and the HF CLI for the videos (pip install huggingface_hub).
#
# Usage:  bash scripts/prepare_train_data.sh [WORKDIR]     (default: ./train_data)
set -euo pipefail
WORKDIR=${1:-train_data}
mkdir -p "$WORKDIR"

# 1) The rewrite JSONs live in the HumanOmniV2 GitHub repo. Each record is one
#    training example (problem, options, answer, and a relative video/image path).
#    convert.py unions the three of them, strips the reasoning traces, and emits the
#    SFT JSONL — that is the only thing these files are used for here. They are NOT
#    used to selectively download videos; the canonical list of the video files you
#    need is the committed data/matched_alldata.jsonl (its 30,217 paths).
if [ ! -d "$WORKDIR/HumanOmniV2" ]; then
    echo "==> Cloning HumanOmniV2 (for the rewrite JSONs)"
    git clone --depth 1 https://github.com/HumanMLLM/HumanOmniV2.git "$WORKDIR/HumanOmniV2"
fi
JSON_DIR="$WORKDIR/HumanOmniV2/src/open-r1-multimodal/data_config"
echo "Rewrite JSONs:  $JSON_DIR"
ls "$JSON_DIR"/{emer_rewrite,social_iq_v2_rewrite,Video-R1_rewrite}.json

cat <<EOF

==> Videos
The 30,217 examples reference videos/images from just three of HumanOmniV2's
component corpora. Get those and arrange them under a single video root so the
relative paths resolve:

  train_data/videos/
    MER24/              (MER2024 / EMER)
    social_iq/          (Social-IQ 2.0)
    Video-R1-data/      (Video-R1, github.com/tulerfeng/Video-R1)

(HumanOmniV2's full corpus also lists AVQA-R1-6K and OmniInstruct, but the traces
we keep never reference them, so you do not need those.) To check your videos cover
what training needs, compare against the paths in data/matched_alldata.jsonl.

Pointers: HumanOmniV2 long-CoT release (hf: PhilipC/IntentTrain); Video-R1
(github.com/tulerfeng/Video-R1). See HumanOmniV2's README (Training > Prepare).

==> Build the training JSONL (once videos are in place):
  python data_prep/convert.py \\
    --config-dir "$JSON_DIR" \\
    --data-root  "$WORKDIR/videos" \\
    --output-dir data

This reproduces data/matched_alldata.jsonl (30,217 examples) with absolute paths.
EOF
