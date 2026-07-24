#!/bin/bash
# Daily-Omni / WorldSense evaluation. Writes eval_results/<dataset>_<RUN_NAME>.json;
# summarize with `python eval/compute_accuracy.py`.
#
# Fill in .env first. Required: DATASET (daily|world), RUN_NAME. Examples:
#   sbatch --export=ALL,DATASET=daily,RUN_NAME=vanilla_lora_e1,PLAIN_SFT=1,LORA_PATH=output/vanilla_lora/checkpoint-XXX eval/eval_benchmarks.slurm
#   sbatch --export=ALL,DATASET=world,RUN_NAME=smoke,MAX_SAMPLES=5 eval/eval_benchmarks.slurm
#
# Optional env: LORA_PATH, MODEL_PATH (override base), PLAIN_SFT=1 (our SFT models),
# HOV2_CKPT=1, SEPARATE_AUDIO=1 (separate-stream = HOv2 protocol; default is interleaved TMRoPE), MAX_SAMPLES.
#
# Scheduler flags are placeholders — set them for your cluster (or pass on the sbatch CLI).
#SBATCH --job-name=eval_benchmarks
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=10:00:00
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
if [ -n "${CLUSTER_MODULES:-}" ]; then module purge && module load ${CLUSTER_MODULES}; fi

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export ENABLE_AUDIO_OUTPUT=False

MODEL=${MODEL_PATH:-$BASE_MODEL}
DATASET=${DATASET:?Must set DATASET=daily or world}
RUN_NAME=${RUN_NAME:?Must set RUN_NAME}
mkdir -p logs eval_results

if [ "$DATASET" = "daily" ]; then
    GT_PATH=$DAILY_QA;  VIDEO_ROOT=$DAILY_VIDEOS
elif [ "$DATASET" = "world" ]; then
    GT_PATH=$WORLD_QA;  VIDEO_ROOT=$WORLD_VIDEOS
else
    echo "ERROR: DATASET must be 'daily' or 'world'"; exit 1
fi

EXTRA_FLAGS=""
[ -n "${LORA_PATH:-}" ]        && EXTRA_FLAGS="$EXTRA_FLAGS --lora-path $LORA_PATH"
[ "${PLAIN_SFT:-0}" = "1" ]    && EXTRA_FLAGS="$EXTRA_FLAGS --plain-sft"
[ "${HOV2_CKPT:-0}" = "1" ]    && EXTRA_FLAGS="$EXTRA_FLAGS --hov2-ckpt"
[ "${SEPARATE_AUDIO:-0}" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --separate-audio-stream"
[ -n "${MAX_SAMPLES:-}" ]      && EXTRA_FLAGS="$EXTRA_FLAGS --max-samples $MAX_SAMPLES"

# HOv2 emits <think>/<answer> traces — a 10-token budget would truncate the
# answer tag and silently zero the score.
MAX_NEW_TOKENS=$([ "${HOV2_CKPT:-0}" = "1" ] && echo 2048 || echo 10)

NGPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
echo "Dataset: $DATASET | Run: $RUN_NAME | Model: $MODEL | GPUs: $NGPUS | flags:$EXTRA_FLAGS"

srun "$PYTHON" -m torch.distributed.run \
    --nproc_per_node "$NGPUS" \
    --master_port $((29500 + SLURM_JOB_ID % 2000)) \
    --nnodes 1 \
    eval/eval_benchmarks.py \
    --model-path "$MODEL" \
    --dataset "$DATASET" \
    --gt-path "$GT_PATH" \
    --video-root "$VIDEO_ROOT" \
    --file-name "$RUN_NAME" \
    --output-dir eval_results \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    $EXTRA_FLAGS

echo "=== Eval complete: $DATASET / $RUN_NAME ==="
