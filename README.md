# Vanilla SFT

Code for the **§4** and **§5** experiments of **"Reasoning for Social Audio-Visual Question
Answering: Where Do We Stand?"** (ECCV 2026 HCMIW workshop submission).

**§4 Vanilla SFT** A trivial baseline — *Vanilla* supervised fine-tuning that outputs only the
answer, no reasoning trace — trained on HumanOmniV2's *own* training data, matches or beats
HumanOmniV2's full chain-of-thought + reinforcement-learning pipeline, and other reasoning methods, across three benchmarks. The
reasoning traces and RL add little beyond the training data itself, at a large cost in latency and
compute. We call this baseline **Vanilla SFT** and argue a similar baseline should be a mandatory point of
comparison for any proposed method in this subfield.

**§5 Leveraging video** A set of three LoRA fine-tunes on the *same*
Social-IQ 2.0 + EMER data, each restricted to a different information channel — **question-only**
(text), **caption-in-prompt** (text), and **audio+video+text** (vanilla) — each compared to the
Qwen2.5-Omni-7B base in that same channel, on IntentBench-Prime (Hard). See
[the §5 section below](#5--when-to-use-reasoning-traces-sft-variants). The two sections are kept
separate throughout the repo: §5 assets live under `*/sft_variants/`.

Base model: **Qwen2.5-Omni-7B**. Framework: **ms-swift**. IntentBench scoring uses our companion
benchmark [**IntentBench-Prime**](https://github.com/koenv759/IntentBench-Prime); a frozen copy
of its (stdlib-only) scorer is vendored as the single self-contained `eval/score_intentbench.py`
(exclusion lists in `eval/ib_data/`), so there is nothing extra to install to reproduce the numbers.

## Results

This repo ships the evaluation result files for four models under `results/` (our two Vanilla-SFT
configs plus the two baselines), so you can reproduce every number below with **no GPU**. All four
tables below are **our own reproductions**, scored from those shipped files:

| Model                            | IB Full | IB Clean | IB Hard | WorldSense | Daily-Omni |
|----------------------------------|:-------:|:--------:|:-------:|:----------:|:----------:|
| Base Qwen2.5-Omni-7B             | 64.58 | 67.83 | 63.22 | 43.69 | 61.07 |
| HumanOmniV2                      | 68.90 | 71.76 | 66.91 | 47.26 | 58.40 |
| **Vanilla SFT — LoRA** (1 ep)    | 70.60 | 73.75 | 70.43 | 48.77 | 65.16 |
| **Vanilla SFT — full-FT** (2 ep) | 70.46 | 73.56 | 69.02 | 46.72 | 62.07 |

*IB = IntentBench; Clean/Hard = the IntentBench-Prime variants (broken-removed / broken+text-answerable-removed).*

**Reproductions vs. the paper.** All numbers above are our own reproductions. In the paper we report
the originating works' published numbers where applicable, so some baseline cells there differ from
these (the IntentBench-Prime Clean/Hard columns have no external source, so those are ours throughout).

**Audio protocol.** The HumanOmniV2 files were produced with its **native separate-stream** audio
protocol (`SEPARATE_AUDIO=1`); base Qwen and our SFT models use interleaved TMRoPE. See
[Key settings](#key-settings-dont-change-these-to-reproduce).

## Install

Scoring the shipped result files needs **nothing installed** — the vendored `eval/score_intentbench.py`
scorer uses only the Python standard library. For training or running evals (needs a GPU):

```bash
pip install -e ".[eval]"  # torch / transformers / ms-swift / accelerate / deepspeed / …
```

Then copy the environment template and fill in your paths:

```bash
cp .env.example .env      # then edit: model, caches, dataset paths, cluster modules
```

## Reproduce the paper numbers, no GPU

All four models' evaluation outputs ship in `results/` (QA text and cluster paths stripped; only the
fields the scorers read are kept), one directory per model.

```bash
# IntentBench Full / Clean / Hard for the LoRA run (-> 70.60 / 73.75 / 70.43):
python eval/score_intentbench.py results/vanilla_lora_e1/ib_vanilla_lora_e1.json

# WorldSense + Daily-Omni per-category tables for everything in a results dir:
python eval/compute_accuracy.py --results-dir results/vanilla_lora_e1
```

## Reproduce from scratch (GPU)

Full walkthrough in [`docs/reproduce.md`](docs/reproduce.md). In short:

```bash
# 1. Data
bash scripts/download_benchmarks.sh          # IntentBench, Daily-Omni, WorldSense
bash scripts/prepare_train_data.sh           # HumanOmniV2 rewrite JSONs + videos -> data/matched_alldata.jsonl
#    (edit .env so the dataset paths point at what you downloaded)

# 2. Train (4x H100 in the paper; set scheduler flags for your cluster)
sbatch --export=ALL,CONFIG=configs/vanilla_lora.yaml   training/train.slurm   # LoRA, 1 epoch
sbatch --export=ALL,CONFIG=configs/vanilla_fullft.yaml training/train.slurm   # full-FT, 2 epochs

# 3. Evaluate (point LORA_PATH at the checkpoint; interleaved TMRoPE audio is the default)
sbatch --export=ALL,RUN_NAME=vanilla_lora_e1,LORA_PATH=output/vanilla_lora/checkpoint-XXX eval/eval_intentbench.slurm
sbatch --export=ALL,DATASET=daily,RUN_NAME=vanilla_lora_e1,PLAIN_SFT=1,LORA_PATH=output/vanilla_lora/checkpoint-XXX eval/eval_benchmarks.slurm
sbatch --export=ALL,DATASET=world,RUN_NAME=vanilla_lora_e1,PLAIN_SFT=1,LORA_PATH=output/vanilla_lora/checkpoint-XXX eval/eval_benchmarks.slurm

# 4. Score
python eval/score_intentbench.py eval_results/ib_vanilla_lora_e1_direct_answer.json
python eval/compute_accuracy.py --results-dir eval_results
```

## §5 — When to use reasoning traces (SFT variants)

A separate experiment from §4. Three LoRA fine-tunes on the **same** Social-IQ 2.0 + EMER data
(2,568 text-only / 2,575 video examples), each seeing a different information channel, each compared
to the Qwen2.5-Omni-7B **base** in that same channel. Evaluated on **IntentBench-Prime (Hard)** only.
Unlike §4's full-data Vanilla SFT, these variants are restricted to SIQ + EMER so the three channels
are directly comparable. Their assets live under `*/sft_variants/`.

| Information channel        | Base (Hard) | SFT variant (Hard) |
|----------------------------|:-----------:|:------------------:|
| Question only (text)       |    46.93    |      62.02         |
| Caption in prompt (text)   |    62.14    |      68.64         |
| Audio + video + text       |    63.22    |      69.52         |

*All numbers are our reproductions, scored from the shipped `results/sft_variants/` fixtures with
`eval/score_intentbench.py` (the paper's §5 table reports the same, rounded to 46.9 / 62.1 / 63.2 /
62.0 / 68.6 / 69.5). The audio+video **base** cell is the §4 `results/base_qwen` run scored on Hard —
the base model is shared, so it is not duplicated. Note the §5 vanilla variant (69.52) is distinct
from §4's full-data Vanilla SFT — LoRA (70.43 Hard); the variant uses only SIQ + EMER.*

**Reproduce the numbers, no GPU:**

```bash
python eval/score_intentbench.py results/sft_variants/ib_question_sft_question.json      # -> Hard 62.02
python eval/score_intentbench.py results/sft_variants/ib_caption_sft_caption.json        # -> Hard 68.64
python eval/score_intentbench.py results/sft_variants/ib_vanilla_sft_direct_answer.json  # -> Hard 69.52
```

**Reproduce from scratch (GPU):** train + evaluate walkthrough in
[`docs/reproduce_sft_variants.md`](docs/reproduce_sft_variants.md). In short — train with
`train_sft_variants.slurm` (`CONFIG` selects the channel), then evaluate `eval_intentbench.py` in the
matching `--modality` (`question` / `caption` / `video`). Caption needs `eval/ib_captions/asid_7B.json`.

## Layout

```
configs/     ms-swift training configs (vanilla_lora.yaml, vanilla_fullft.yaml) + fix_lr_plugin.py
configs/sft_variants/  §5 configs (question.yaml, caption.yaml, vanilla.yaml)
data_prep/   convert.py — HumanOmniV2 rewrite JSONs -> plain-answer SFT JSONL
data/        matched_alldata.jsonl — the exact §4 training set (traces stripped, relative video paths)
data/sft_variants/     §5 training sets (question / caption text-only; vanilla with videos)
training/    train.slurm (§4) + train_sft_variants.slurm (§5) — parametrized launchers (CONFIG selects the run)
eval/        eval_intentbench.py (--modality video|caption|question|both), eval_benchmarks.py, compute_accuracy.py + slurms
eval/score_intentbench.py  vendored IntentBench-Prime scorer (single file, stdlib only)
eval/ib_data/      IntentBench-Prime variant manifest + exclusion lists (JSON)
eval/ib_captions/  asid_7B.json — ASID captions for the §5 caption/both settings
results/     §4 outputs (base_qwen, humanomniv2, vanilla_lora_e1, vanilla_fullft_e2), one dir per model
results/sft_variants/  §5 outputs (question/caption base + the 3 SFT-variant fixtures)
scripts/     download_benchmarks.sh, prepare_train_data.sh
docs/        reproduce.md (§4), reproduce_sft_variants.md (§5)
```

## Key settings

- **Frame sampling: FPS = 2, max 32 frames** — the training-and-eval setting throughout (also HumanOmniV2's).
- **LoRA: report epoch 1. Full-FT: report epoch 2.** Further LoRA training degrades.
- **Audio: interleaved TMRoPE** (`USE_AUDIO_IN_VIDEO=True` in training) — the default for our models
  and base Qwen at eval time. To reproduce HumanOmniV2's separate-stream protocol instead, pass
  `SEPARATE_AUDIO=1` (`--separate-audio-stream`).
- **Direct-answer prompting** for SFT models (`PLAIN_SFT=1`); the eval scripts also have a `--hov2-ckpt`
  mode for the HumanOmniV2 reasoning-model comparison.

## Citation

> *Reasoning for Social Audio-Visual Question Answering: Where Do We Stand?* (submitted to ECCV HCMIW 2026).

<!-- TODO: BibTeX once the citation is final. -->

## Acknowledgements

Built on [**HumanOmniV2**](https://github.com/HumanMLLM/HumanOmniV2), whose training data we reuse
(a re-formatted subset of Social-IQ 2.0 / EMER / Video-R1 / OmniInstruct) and whose IntentBench eval
script `eval/eval_intentbench.py` is adapted from. Base model:
[**Qwen2.5-Omni-7B**](https://github.com/QwenLM/Qwen2.5-Omni); training via
[**ms-swift**](https://github.com/modelscope/ms-swift).

## License

The **code** is under the **GNU GPL v3.0 or later** (see [`LICENSE`](LICENSE)).
`eval/eval_intentbench.py` is adapted from HumanOmniV2's Apache-2.0 eval script (Apache-2.0 is
GPLv3-compatible; origin noted in-file).

The **data is not redistributed here** — training JSONLs carry only relative video paths, and
`results/` files are QA-stripped. The videos come from HumanOmniV2 and its upstream sources (its
self-collected videos are **CC BY-NC-SA 4.0**, non-commercial); obtain them there (see
[`docs/reproduce.md`](docs/reproduce.md)).
