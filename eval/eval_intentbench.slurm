#!/bin/bash
# IntentBench evaluation (DataLoader-accelerated, direct-answer + reasoning modes).
# Writes eval_results/ib_<RUN_NAME>_<mode>.json — score it with the vendored scorer
# (`python eval/score_intentbench.py <file>`), or use the --latency table below.
#
# Fill in .env first. Required: RUN_NAME. Examples:
#   # Our LoRA model (interleaved audio = default), full IntentBench:
#   sbatch --export=ALL,RUN_NAME=vanilla_lora_e1,LORA_PATH=output/vanilla_lora/checkpoint-XXX eval/eval_intentbench.slurm
#   # Smoke test (20 samples, 1 GPU):
#   sbatch --gres=gpu:1 --cpus-per-task=24 --export=ALL,RUN_NAME=smoke,MAX_SAMPLES=20 eval/eval_intentbench.slurm
#   # Latency subset (IntentBench-hard, fixed random 300 on 1 GPU, batch=1):
#   sbatch --gres=gpu:1 --cpus-per-task=24 --export=ALL,RUN_NAME=lora_lat,LATENCY=1,LORA_PATH=...,EXCLUDE_QIDS=hard,SHUFFLE_SEED=0,MAX_SAMPLES=300 eval/eval_intentbench.slurm
#
# Optional env: LORA_PATH, MODEL_PATH (override base), SEPARATE_AUDIO=1 (separate-stream
# = HOv2 protocol; default is interleaved TMRoPE), HOV2_CKPT=1, MAX_SAMPLES,
# LATENCY=1, SHUFFLE_SEED, EXCLUDE_QIDS ('clean'|'hard' via the vendored scorer, or a qid-JSON path).
#
# §5 SFT variants (information channels). MODALITY = video (default) | caption | question | both;
# caption/both also need CAPTION_JSON. Writes ib_<RUN_NAME>_<modality>.json (video -> _direct_answer):
#   # Question-only SFT model (text-only, no media):
#   sbatch --export=ALL,RUN_NAME=question_sft,MODALITY=question,LORA_PATH=output/sft_variants/question/checkpoint-XXX eval/eval_intentbench.slurm
#   # Caption-in-prompt SFT model (ASID caption in prompt, no video):
#   sbatch --export=ALL,RUN_NAME=caption_sft,MODALITY=caption,CAPTION_JSON=eval/ib_captions/asid_7B.json,LORA_PATH=output/sft_variants/caption/checkpoint-XXX eval/eval_intentbench.slurm
#
# Scheduler flags are placeholders — set them for your cluster (or pass on the sbatch CLI).
#SBATCH --job-name=eval_ib
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
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
RUN_NAME=${RUN_NAME:?Must set RUN_NAME}
mkdir -p logs eval_results

EXTRA_FLAGS=""
[ -n "${LORA_PATH:-}" ]        && EXTRA_FLAGS="$EXTRA_FLAGS --lora-path $LORA_PATH"
[ "${HOV2_CKPT:-0}" = "1" ]    && EXTRA_FLAGS="$EXTRA_FLAGS --hov2-ckpt"
[ "${SEPARATE_AUDIO:-0}" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --separate-audio-stream"
[ -n "${MAX_SAMPLES:-}" ]      && EXTRA_FLAGS="$EXTRA_FLAGS --max-samples $MAX_SAMPLES"
[ "${LATENCY:-0}" = "1" ]      && EXTRA_FLAGS="$EXTRA_FLAGS --latency"
[ -n "${SHUFFLE_SEED:-}" ]     && EXTRA_FLAGS="$EXTRA_FLAGS --shuffle-seed $SHUFFLE_SEED"
[ -n "${EXCLUDE_QIDS:-}" ]     && EXTRA_FLAGS="$EXTRA_FLAGS --exclude-qids $EXCLUDE_QIDS"
[ -n "${MODALITY:-}" ]         && EXTRA_FLAGS="$EXTRA_FLAGS --modality $MODALITY"
[ -n "${CAPTION_JSON:-}" ]     && EXTRA_FLAGS="$EXTRA_FLAGS --caption-json $CAPTION_JSON"

NGPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}
echo "Run: $RUN_NAME | Model: $MODEL | GPUs: $NGPUS | flags:$EXTRA_FLAGS"

srun "$PYTHON" -m torch.distributed.run \
    --nproc_per_node "$NGPUS" \
    --master_port $((29600 + SLURM_JOB_ID % 2000)) \
    --nnodes 1 \
    eval/eval_intentbench.py \
    --base-model-path "$MODEL" \
    --ib-path "$IB_QA" \
    --video-root "$IB_VIDEOS" \
    --results-dir eval_results \
    --run-name "$RUN_NAME" \
    --fps 2 --max-frames 32 \
    $EXTRA_FLAGS

echo "=== IntentBench eval complete: $RUN_NAME ==="
