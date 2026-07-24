#!/bin/bash
# SFT-variants training (ms-swift) — the paper's §5 runs (Social-IQ 2.0 + EMER only).
#
# Three information-channel LoRA fine-tunes on the SAME data, selected via CONFIG:
#   sbatch --export=ALL,CONFIG=configs/sft_variants/question.yaml training/train_sft_variants.slurm  # text: question only
#   sbatch --export=ALL,CONFIG=configs/sft_variants/caption.yaml  training/train_sft_variants.slurm  # text: ASID caption in prompt
#   sbatch --export=ALL,CONFIG=configs/sft_variants/vanilla.yaml  training/train_sft_variants.slurm  # audio+video+text
# All three train 3 epochs; report the epoch-3 checkpoint (output/sft_variants/*/…/checkpoint of epoch 3).
#
# NB — frame sampling differs from §4. These §5 runs used FPS=1 / 128 frames (only the
# vanilla run reads video at all; the two text-only runs ignore it). §4's Vanilla SFT
# uses FPS=2 / 32. IntentBench *evaluation* is FPS=2 / 32 for both sections.
#
# The paper runs used 4x H100. Scheduler flags below are placeholders — set them for
# your cluster (or pass on the sbatch CLI, e.g. --account=... --qos=... --constraint=...).
#SBATCH --job-name=sft_variants
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --hint=nomultithread
## --- Cluster-specific scheduler flags: uncomment and set for your site ---
##SBATCH --account=CHANGE_ME
##SBATCH --qos=CHANGE_ME
##SBATCH --constraint=CHANGE_ME

set -euo pipefail
cd "$(dirname "$0")/.."          # repo root
source .env

# Optional cluster module loads (from .env CLUSTER_MODULES; no-op if empty/unset).
if [ -n "${CLUSTER_MODULES:-}" ]; then module purge && module load ${CLUSTER_MODULES}; fi

CONFIG=${CONFIG:-configs/sft_variants/question.yaml}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export ENABLE_AUDIO_OUTPUT=False
export USE_AUDIO_IN_VIDEO=True   # interleaved TMRoPE audio (the training protocol)
export FPS=1                     # §5 frame sampling (only the vanilla variant reads video)
export FPS_MAX_FRAMES=128
export VIDEO_MAX_TOKEN_NUM=128

mkdir -p logs

NGPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
echo "Config: $CONFIG | Model: $BASE_MODEL | GPUs: $NGPUS"

NPROC_PER_NODE=$NGPUS srun "$SWIFT_BIN" sft --config "$CONFIG" --model "$BASE_MODEL"

echo "=== SFT variant complete: $CONFIG ==="
