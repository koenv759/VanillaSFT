# Reproducing the §4 (Vanilla SFT) results

End-to-end: environment → data → train → evaluate → score. Training and evaluation need GPUs
(the paper used 4× H100); scoring existing result files does not.

## 0. Environment

```bash
pip install -e ".[eval]"     # torch/transformers/ms-swift/… (the scorer is vendored, needs no install)
cp .env.example .env         # then edit (see fields below)
```

`.env` fields (sourced by the SLURM scripts):

| Var | Meaning |
|-----|---------|
| `PYTHON` / `SWIFT_BIN` | interpreter for eval / the ms-swift `swift` entrypoint for training |
| `BASE_MODEL` | `Qwen/Qwen2.5-Omni-7B` (auto-downloaded) or a local snapshot dir |
| `HF_HOME` | Hugging Face cache location |
| `IB_QA` / `IB_VIDEOS` | IntentBench `qa.json` + videos dir |
| `DAILY_QA` / `DAILY_VIDEOS`, `WORLD_QA` / `WORLD_VIDEOS` | benchmark QA + video roots |
| `CLUSTER_MODULES` | optional `module load` list for your site (e.g. `arch/h100 anaconda-py3 ffmpeg`) |

The `#SBATCH --account/--qos/--constraint` lines in the `.slurm` files are placeholders —
uncomment and set them for your scheduler, or pass them on the `sbatch` command line.

## 1. Data

```bash
bash scripts/download_benchmarks.sh    # IntentBench, Daily-Omni, WorldSense (HF datasets)
bash scripts/prepare_train_data.sh     # HumanOmniV2 rewrite JSONs (+ pointers to the videos)
```

- **Benchmarks** come from `PhilipC/IntentBench`, `liarliar/Daily-Omni`, `honglyhly/WorldSense`.
  Extract any video archives and set the `*_QA` / `*_VIDEOS` paths in `.env` accordingly.
- **Training data.** `data/matched_alldata.jsonl` (committed) already defines the exact 30,217
  training examples — HumanOmniV2's data with the reasoning traces removed — using repo-relative
  video paths (`videos/MER24/…`, `videos/social_iq/…`, `videos/Video-R1-data/…`). Place the videos
  under `videos/`, **or** regenerate the JSONL with absolute paths to wherever you put them:

  ```bash
  python data_prep/convert.py --config-dir <HumanOmniV2>/src/open-r1-multimodal/data_config \
                              --data-root  <your_videos_root> --output-dir data
  ```

## 2. Train

```bash
# LoRA — the headline setting (1 epoch, IB 70.60):
sbatch --export=ALL,CONFIG=configs/vanilla_lora.yaml   training/train.slurm
# Full fine-tune (2 epochs, IB 70.46):
sbatch --export=ALL,CONFIG=configs/vanilla_fullft.yaml training/train.slurm
```

Both freeze the vision encoder, train at FPS 2 / max 32 frames with interleaved audio, effective
batch 64. LoRA is rank 16, lr 1e-4; full-FT is lr 2e-6 (zero2_offload — zero3 hangs with
qwen2_5_omni). The paper runs finished in <4.5 h on 4× H100. Checkpoints land in
`output/vanilla_lora/` and `output/vanilla_fullft/` (one per epoch).

**ms-swift does not auto-resume** a timed-out run pointed at an existing `output_dir` — it restarts
from step 1. To continue, add `resume_from_checkpoint: <ckpt-dir>` to the config (and remove it again
before the next fresh run).

## 3. Evaluate

Point `LORA_PATH` (LoRA) or `MODEL_PATH` (full-FT dir) at the checkpoint. Interleaved TMRoPE audio
(our training protocol) is the default; use `PLAIN_SFT=1` (direct-answer prompt) for our models.

```bash
# IntentBench:
sbatch --export=ALL,RUN_NAME=vanilla_lora_e1,LORA_PATH=output/vanilla_lora/checkpoint-XXX eval/eval_intentbench.slurm
# WorldSense / Daily-Omni:
sbatch --export=ALL,DATASET=world,RUN_NAME=vanilla_lora_e1,PLAIN_SFT=1,LORA_PATH=output/vanilla_lora/checkpoint-XXX eval/eval_benchmarks.slurm
sbatch --export=ALL,DATASET=daily,RUN_NAME=vanilla_lora_e1,PLAIN_SFT=1,LORA_PATH=output/vanilla_lora/checkpoint-XXX eval/eval_benchmarks.slurm
```

Outputs: `eval_results/ib_<RUN_NAME>_direct_answer.json`, `eval_results/{world,daily}_<RUN_NAME>.json`.

The HumanOmniV2 comparison uses the same scripts with `HOV2_CKPT=1` (thinker prompt + `<answer>`
tags, generation budget raised to 2048 tokens). To evaluate it in HumanOmniV2's own separate-stream
audio protocol, add `SEPARATE_AUDIO=1`.

**Latency table (Experiment 1).** Run IntentBench in `--latency` mode on **1 GPU, batch 1**, over a
fixed random subset of the IntentBench-hard variant:

```bash
sbatch --gres=gpu:1 --cpus-per-task=24 \
  --export=ALL,RUN_NAME=lora_lat,LATENCY=1,LORA_PATH=output/vanilla_lora/checkpoint-XXX,EXCLUDE_QIDS=hard,SHUFFLE_SEED=0,MAX_SAMPLES=300 \
  eval/eval_intentbench.slurm
```

`EXCLUDE_QIDS=hard` resolves the 790-qid IntentBench-hard exclusion set from the vendored
`eval/score_intentbench.py` scorer (no separate install needed). Writes `ib_<RUN_NAME>_direct_answer_latency.json`.

## 4. Score

```bash
# IntentBench -> Full / Clean / Hard (IntentBench-Prime):
python eval/score_intentbench.py eval_results/ib_vanilla_lora_e1_direct_answer.json

# WorldSense + Daily-Omni per-category tables + results_summary.xlsx:
python eval/compute_accuracy.py --results-dir eval_results
```

`score_intentbench.py` is the vendored IntentBench-Prime scorer (a single self-contained file); it
filters your IntentBench result file to the Clean/Hard variants and recomputes accuracy (no
re-evaluation needed). Equivalently, in Python from the `eval/` dir:
`from score_intentbench import accuracy; accuracy(results, "hard")`.
