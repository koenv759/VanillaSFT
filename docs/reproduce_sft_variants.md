# Reproducing the §5 (SFT variants) results

The §5 experiment asks *which information channel the SFT gain actually comes from*. Three LoRA
fine-tunes share the **same** Social-IQ 2.0 + EMER training data but see a different channel:

| Variant           | Config                            | Channel (train & eval)          | Data |
|-------------------|-----------------------------------|---------------------------------|------|
| Question-only     | `configs/sft_variants/question.yaml` | question + options, text only | `data/sft_variants/question.jsonl` (2,568) |
| Caption-in-prompt | `configs/sft_variants/caption.yaml`  | ASID caption in prompt, text  | `data/sft_variants/caption.jsonl` (2,568) |
| Vanilla           | `configs/sft_variants/vanilla.yaml`  | audio + video + text          | `data/sft_variants/vanilla.jsonl` (2,575) |

Each is compared to the Qwen2.5-Omni-7B **base** in the same channel, on **IntentBench-Prime (Hard)**.
Scoring the shipped `results/sft_variants/` fixtures needs no GPU (see the README §5 table); training
and re-evaluating do. Environment setup is identical to [`reproduce.md`](reproduce.md) §0.

## 1. Data

The three JSONLs are committed under `data/sft_variants/` — already built, with repo-relative video
paths (`videos/social_iq/…`, `videos/MER24/…`) for the vanilla run. They were derived from
Social-IQ 2.0 + EMER: the question set keeps only the question + options; the caption set pastes a
pre-generated **ASID** caption (Qwen2.5-Omni-7B captioner, `7B/default`) into the prompt in place of
the video; the vanilla set is the plain `<video>`+question → answer form. The ASID captioner itself
is **not** shipped; the captions used at *eval* time are in `eval/ib_captions/asid_7B.json` (633
IntentBench videos). Place the SIQ + EMER videos under `videos/` for the vanilla run (the two
text-only runs need no videos).

## 2. Train

One launcher, `CONFIG` selects the channel (3 epochs each; report the **epoch-3** checkpoint):

```bash
sbatch --export=ALL,CONFIG=configs/sft_variants/question.yaml training/train_sft_variants.slurm
sbatch --export=ALL,CONFIG=configs/sft_variants/caption.yaml  training/train_sft_variants.slurm
sbatch --export=ALL,CONFIG=configs/sft_variants/vanilla.yaml  training/train_sft_variants.slurm
```

Same LoRA recipe as §4 (rank 16, α 32, lr 1e-4, all-linear, freeze_vit, zero2, interleaved TMRoPE
audio). **Frame sampling differs from §4:** these runs use `FPS=1` / 128 frames (set in
`train_sft_variants.slurm`); only the vanilla run reads video at all. IntentBench *evaluation* is
`FPS=2` / 32 frames for both sections. Checkpoints land in `output/sft_variants/{question,caption,vanilla}/`.

## 3. Evaluate

Evaluate each model in the channel it was trained on, via `--modality` (the SLURM `MODALITY` env):

```bash
# Question-only (text-only, no media):
sbatch --export=ALL,RUN_NAME=question_sft,MODALITY=question,LORA_PATH=output/sft_variants/question/checkpoint-XXX eval/eval_intentbench.slurm
# Caption-in-prompt (ASID caption in prompt, no video):
sbatch --export=ALL,RUN_NAME=caption_sft,MODALITY=caption,CAPTION_JSON=eval/ib_captions/asid_7B.json,LORA_PATH=output/sft_variants/caption/checkpoint-XXX eval/eval_intentbench.slurm
# Vanilla (audio+video+text = the default modality):
sbatch --export=ALL,RUN_NAME=vanilla_sft,LORA_PATH=output/sft_variants/vanilla/checkpoint-XXX eval/eval_intentbench.slurm
```

For the **base** rows, run the same commands with no `LORA_PATH` (base Qwen). The audio+video base is
just the §4 `results/base_qwen/ib_base_qwen.json` run scored on Hard — no need to re-run it.

Outputs: `eval_results/ib_<RUN_NAME>_<modality>.json` (the video channel uses the `_direct_answer`
suffix, as in §4).

## 4. Score

The paper's §5 table is **Hard** only:

```bash
python eval/score_intentbench.py eval_results/ib_question_sft_question.json      # Hard column
python eval/score_intentbench.py eval_results/ib_caption_sft_caption.json
python eval/score_intentbench.py eval_results/ib_vanilla_sft_direct_answer.json
```

`score_intentbench.py` prints Full / Clean / Hard side by side; §5 reports Hard.
