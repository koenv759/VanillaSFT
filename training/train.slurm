#!/bin/bash
# Vanilla SFT training (ms-swift) — the paper's §4 runs.
#
# Fill in .env (copy from .env.example), then submit, choosing the config via CONFIG:
#   sbatch --export=ALL,CONFIG=configs/vanilla_lora.yaml   training/train.slurm   # LoRA, 1 epoch  -> IB 70.60
#   sbatch --export=ALL,CONFIG=configs/vanilla_fullft.yaml training/train.slurm   # full-FT, 2 ep. -> IB 70.46
#
# The paper runs used 4x H100. Scheduler flags below are placeholders — set them for
# your cluster (or pass on the sbatch CLI, e.g. --account=... --qos=... --constraint=...).
#SBATCH --job-name=vanilla_sft
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=14:00:00
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

CONFIG=${CONFIG:-configs/vanilla_lora.yaml}

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export ENABLE_AUDIO_OUTPUT=False
export USE_AUDIO_IN_VIDEO=True   # interleaved TMRoPE audio (the training protocol)
export FPS=2                     # frame sampling — THE setting (matches eval + HOv2)
export FPS_MAX_FRAMES=32
export VIDEO_MAX_TOKEN_NUM=128

mkdir -p logs

NGPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
echo "Config: $CONFIG | Model: $BASE_MODEL | GPUs: $NGPUS"

NPROC_PER_NODE=$NGPUS srun "$SWIFT_BIN" sft --config "$CONFIG" --model "$BASE_MODEL"

echo "=== Vanilla SFT complete: $CONFIG ==="
