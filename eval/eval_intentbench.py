# SPDX-License-Identifier: GPL-3.0-or-later
# Adapted from HumanOmniV2's IntentBench eval script (Apache-2.0).
"""
IntentBench evaluation — DataLoader-accelerated, with four information settings.

process_mm_info (video decode) runs in DataLoader worker processes
(num_workers=4), overlapping with GPU inference.

Information settings (`--modality`, one result file each):
  video     Answer from video + question (the §4 direct-answer path). DEFAULT.
  caption   Answer from an ASID caption pasted into the prompt, no video/audio
            attached ("Here is a description of a video: [CAPTION] …"). Text-only.
  both      Caption in the prompt AND the video attached.
  question  Only the question + options — no video, no caption. Text-only.
            Isolates the learned dataset/answer prior.
Caption/both require --caption-json (e.g. eval/ib_captions/asid_7B.json).
The video-only setting is §4; caption/question/both back the §5 SFT-variants
experiment. Score any output with `python eval/score_intentbench.py <file>`.

Output schema: {qid, video, output, solution, problem_type, raw, load_error}.

Protocol flags for the video-containing settings (video / both):
  default            interleaved audio (TMRoPE): audio extracted from the video
                     itself, no text prefix — matches our training data
                     (bare <video>question). Used for our SFT models and base Qwen.
  --separate-audio-stream
                     separate-stream audio (HOv2 protocol: explicit audio element +
                     "Here is a video..." prefix, use_audio_in_video=False). Pass
                     this to reproduce HumanOmniV2's own evaluation protocol.
  --hov2-ckpt        HOv2 thinker system prompt + their per-type <answer>-tag
                     templates + max_new_tokens 2048 — ONLY for the HOv2 checkpoint
                     (video modality; pass its dir as --base-model-path, no LoRA)

Usage (Jean-Zay, N GPUs):
  torchrun --nproc_per_node N eval/eval_intentbench.py \
      --base-model-path <path> --ib-path <qa.json> \
      --video-root <dir> --run-name <tag> [--modality caption \
      --caption-json eval/ib_captions/asid_7B.json]
"""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import argparse
import itertools
import json
import random
import re
import statistics
import sys
import tempfile
import time

import torch
from peft import PeftModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (AutoConfig, LogitsProcessor, LogitsProcessorList,
                          Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor)
from qwen_omni_utils import process_mm_info
from safetensors.torch import load_file, save_file
import av


# ── prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a helpful assistant."

QA_SUFFIX = " Provide only the single letter corresponding to the correct answer:"

EMER_SUFFIX = (
    " Multiple answers may be correct. Provide the letter(s) of the "
    "correct answer(s), separated by commas:"
)

# Separate-stream prefix — copied verbatim from HOv2 eval_humanomniv2.py.
AUDIO_PREFIX = "Here is a video, with the audio from the video.\n"

# HOv2 thinker prompt + per-type suffixes — copied verbatim from their
# eval_humanomniv2.py (SYSTEM_PROMPT + TYPE_TEMPLATE). Used only with --hov2-ckpt.
THINKER_SYSTEM_PROMPT = (
    "You are a helpful assistant. Your primary goal is to deeply analyze and interpret "
    "information from available various modalities (image, video, audio, text context) "
    "to answer questions with human-like depth and a clear, traceable thought process.\n\n"
    "Begin by thoroughly understanding the image, video, audio or other available context "
    "information, and then proceed with an in-depth analysis related to the question. \n\n"
    "In reasoning, It is encouraged to incorporate self-reflection and verification into "
    "your reasoning process. You are encouraged to review the image, video, audio, or other "
    "context information to ensure the answer accuracy.\n\n"
    "Provide your understanding of the image, video, and audio between the <context> </context> "
    "tags, detail the reasoning between the <think> </think> tags, and then give your final "
    "answer between the <answer> </answer> tags.\n"
)

THINKER_TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "emer_ov_mc":      " Please provide only the single or multiple option letter (e.g., A for single option or A,E for multi option, etc.) within the <answer> </answer> tags.",
    "numerical":       " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "regression":      " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "free-form":       " Please provide your text answer within the <answer> </answer> tags.",
    "OCR":             " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "judge":           " Please answer Yes or No within the <answer> </answer> tags.",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def check_audio(video_path):
    try:
        container = av.open(video_path)
        return any(s.type == "audio" for s in container.streams)
    except Exception:
        return False


def format_question(sample):
    q = sample["problem"]
    if sample["problem_type"] in ("multiple choice", "emer_ov_mc"):
        q += " Options:\n" + "".join(opt + "\n" for opt in sample["options"])
    elif sample["problem_type"] == "judge":
        q += " Options:\nA. Yes\nB. No\n"
    return q


def standard_qa_text(sample):
    """Direct-answer wording (video / question modality, non-HOv2)."""
    q = format_question(sample)
    if sample["problem_type"] == "emer_ov_mc":
        return "Answer the following emotion recognition question." + EMER_SUFFIX + "\n" + q
    return "Answer the following question." + QA_SUFFIX + "\n" + q


def caption_qa_text(sample, modality, caption):
    """Caption-in-prompt wording (caption / both modalities). Mirrors the
    caption-in-prompt SFT template: the ASID caption is handed to the model as
    prompt context. `both` also cues that a video is present."""
    q = format_question(sample)
    prefix = f"Here is a description of a video:\n\n{caption}\n\n"
    base = ("Based on the description, " if modality == "caption"
            else "Based on the video and the description above, ")
    if sample["problem_type"] == "emer_ov_mc":
        lead = base + "answer the following emotion recognition question." + EMER_SUFFIX
    else:
        lead = base + "answer the following question." + QA_SUFFIX
    return prefix + lead + "\n" + q


def extract_answer(text):
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    letters = re.findall(r'\b([A-F])\b', text)
    if letters:
        return ','.join(dict.fromkeys(letters))
    return ''


def reward_fn(pred, gt, problem_type):
    if problem_type == "multiple choice":
        return 1.0 if pred.strip() == gt.strip() else 0.0
    elif problem_type == "judge":
        # Accept both letter answers (direct-answer mode: A=Yes, B=No) and
        # Yes/No text (HOv2 thinker mode — their own scorer is substring-based).
        def _norm(x):
            x = x.strip().lower()
            if "yes" in x:
                return "a"
            if "no" in x:
                return "b"
            return x
        return 1.0 if _norm(pred) == _norm(gt) else 0.0
    elif problem_type == "emer_ov_mc":
        a = set(pred.strip().split(","))
        b = set(gt.strip().split(","))
        tp = len(a & b)
        prec = tp / len(a) if a else 0.0
        rec = tp / len(b) if b else 0.0
        return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return 0.0


# ── latency instrumentation ───────────────────────────────────────────────────

class _FirstTokenTimer(LogitsProcessor):
    """Pass-through logits processor that records a CUDA-synchronised timestamp
    the first time it is called. In HF `generate`, logits processors run once per
    generation step on the step's logits; the first call therefore fires right
    after the prefill forward (vision/audio encoders + LM prefill) has produced
    the first-token logits. torch.cuda.synchronize() forces that async work to
    complete so the timestamp reflects real prefill end (time-to-first-token).
    Fresh instance per sample. Returns scores unchanged → generation output and
    accuracy are identical to a non-latency run.
    """

    def __init__(self):
        self.t_first = None

    def __call__(self, input_ids, scores):
        if self.t_first is None:
            torch.cuda.synchronize()
            self.t_first = time.perf_counter()
        return scores


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {
        "n": len(xs),
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "min": min(xs),
        "max": max(xs),
    }


def summarize_latency(records, wall_clock_s, config):
    """Aggregate per-sample latency records (warmup excluded) into a summary."""
    live = [r for r in records if not r.get("warmup") and not r.get("load_error")]
    dtps = [
        (r["output_tokens"] - 1) / r["decode_s"]
        for r in live
        if r["decode_s"] and r["output_tokens"] and r["output_tokens"] > 1
    ]
    return {
        "config": config,
        "wall_clock_total_s": wall_clock_s,
        "n_timed": len(live),
        "n_warmup_skipped": sum(1 for r in records if r.get("warmup")),
        "output_tokens": _stats([r["output_tokens"] for r in live]),
        "input_tokens": _stats([r["input_tokens"] for r in live]),
        "prefill_s": _stats([r["prefill_s"] for r in live]),
        "decode_s": _stats([r["decode_s"] for r in live]),
        "total_s": _stats([r["total_s"] for r in live]),
        "decode_tokens_per_s": _stats(dtps),
    }


# ── dataset ───────────────────────────────────────────────────────────────────

_MAX_VIDEO_BYTES = 80 * 1024 * 1024


class IBDirectDataset(Dataset):
    """IntentBench direct-answer dataset.

    __getitem__ calls process_mm_info so video decode runs in DataLoader workers,
    overlapping with GPU inference on the previous item.
    """

    def __init__(self, ib_path, video_root, caption_json=None, modality="video",
                 audio_enabled=True, audio_in_video=False,
                 hov2_ckpt=False, fps=2, max_frames=32):
        with open(ib_path) as f:
            self.data = json.load(f)
        self.video_root = video_root
        self.modality = modality
        self.captions = {}
        if caption_json:
            with open(caption_json) as f:
                self.captions = json.load(f)["captions"]
        self.audio_enabled = audio_enabled
        self.audio_in_video = audio_in_video
        self.hov2_ckpt = hov2_ckpt
        self.fps = fps
        self.max_frames = max_frames

    def __len__(self):
        return len(self.data)

    def _load_error(self, qid, video_name, sample, problem_type):
        return {
            "qid": qid, "video": video_name,
            "solution": sample["solution"], "problem_type": problem_type,
            "raw": sample, "load_error": True,
            "images": None, "audios": None, "videos": None,
            "prompt": None, "use_audio_in_video": False,
        }

    def __getitem__(self, index):
        sample = self.data[index]
        video_name = sample.get("video", sample.get("path", ""))
        video_path = os.path.join(self.video_root, video_name)
        qid = sample.get("qid", str(index))
        problem_type = sample["problem_type"]

        needs_video = self.modality in ("video", "both")
        needs_caption = self.modality in ("caption", "both")

        if needs_video and (not os.path.exists(video_path)
                            or os.path.getsize(video_path) > _MAX_VIDEO_BYTES):
            return self._load_error(qid, video_name, sample, problem_type)

        caption = self.captions.get(video_name) if needs_caption else None
        if needs_caption and not caption:
            return self._load_error(qid, video_name, sample, problem_type)

        has_audio = (check_audio(video_path) and self.audio_enabled) if needs_video else False

        # ── build the question text ───────────────────────────────────────────
        if self.modality == "video" and self.hov2_ckpt:
            # HOv2's own question construction (eval_humanomniv2.py): options
            # appended only for MC-style types, then their per-type suffix.
            # Note: no A/B options for judge — their template asks Yes/No.
            qa_text = sample["problem"]
            if problem_type in ("multiple choice", "emer_ov_mc"):
                qa_text += " Options:\n" + "".join(opt + "\n" for opt in sample["options"])
            qa_text += THINKER_TYPE_TEMPLATE[problem_type]
            system_prompt = THINKER_SYSTEM_PROMPT
        elif needs_caption:
            qa_text = caption_qa_text(sample, self.modality, caption)
            system_prompt = SYSTEM_PROMPT
        else:
            qa_text = standard_qa_text(sample)
            system_prompt = SYSTEM_PROMPT

        # ── build the user content / media ────────────────────────────────────
        if self.modality in ("caption", "question"):
            # Text-only: no video, no audio, no decode.
            user_content = [{"type": "text", "text": qa_text}]
            use_aiv = False
            audios, images, videos = None, None, None
        else:
            video_el = {"type": "video", "video": video_path, "fps": self.fps, "max_frames": self.max_frames}
            if self.audio_in_video:
                # Interleaved (TMRoPE): audio extracted from the video itself,
                # no separate element, no text prefix.
                user_content = [video_el, {"type": "text", "text": qa_text}]
                use_aiv = has_audio
            elif has_audio:
                # Separate-stream (HOv2 protocol): explicit audio element + prefix,
                # use_audio_in_video=False everywhere.
                user_content = [
                    video_el,
                    {"type": "audio", "audio": video_path},
                    {"type": "text", "text": AUDIO_PREFIX + qa_text},
                ]
                use_aiv = False
            else:
                user_content = [video_el, {"type": "text", "text": qa_text}]
                use_aiv = False

        message = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user",   "content": user_content},
        ]

        if self.modality not in ("caption", "question"):
            audios, images, videos = process_mm_info(message, use_audio_in_video=use_aiv)

        return {
            "qid": qid, "video": video_name,
            "solution": sample["solution"], "problem_type": problem_type,
            "raw": sample, "load_error": False,
            "images": images, "audios": audios, "videos": videos,
            "prompt": message, "use_audio_in_video": use_aiv,
        }


# ── distributed helpers ───────────────────────────────────────────────────────

class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size, skip=0):
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = [i for i in range(size) if i % self._world_size == self._rank]
        self._skip = skip

    def __iter__(self):
        yield from self._local_indices[self._skip:]

    def __len__(self):
        return len(self._local_indices) - self._skip


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(base_model_path, lora_path):
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    # Two config shapes must both load. A FULL qwen2_5_omni config.json (thinker_config/
    # talker_config/token2wav_config), as some Omni checkpoints ship even when the
    # weights are thinker-only, makes the Thinker's config_class narrow the audio
    # encoder wrongly under recent transformers -> size-mismatch crash; so we parse
    # via AutoConfig and hand from_pretrained the correctly-built thinker sub-config.
    # HOv2 / our SFT checkpoints instead carry model_type "qwen2_5_omni_thinker",
    # which is NOT in transformers' CONFIG_MAPPING -> AutoConfig raises. Fall back to
    # config=None so from_pretrained loads the checkpoint's own (already-thinker) config.
    try:
        _cfg = AutoConfig.from_pretrained(base_model_path)
        thinker_cfg = getattr(_cfg, "thinker_config", None)
    except (KeyError, ValueError):
        thinker_cfg = None
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        base_model_path,
        config=thinker_cfg,
        torch_dtype=torch.bfloat16,
        device_map={"": local_rank},
        attn_implementation="sdpa",
    )
    if lora_path:
        print(f"Loading LoRA adapter from {lora_path} (remapping thinker. prefix) ...")
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(lora_path, "adapter_config.json")) as f:
                cfg = json.load(f)
            cfg["target_modules"] = cfg["target_modules"].replace("thinker\\.", "")
            cfg["base_model_name_or_path"] = base_model_path
            with open(os.path.join(tmp, "adapter_config.json"), "w") as f:
                json.dump(cfg, f)
            weights = load_file(os.path.join(lora_path, "adapter_model.safetensors"))
            remapped = {
                k.replace("base_model.model.thinker.", "base_model.model."): v
                for k, v in weights.items()
            }
            save_file(remapped, os.path.join(tmp, "adapter_model.safetensors"))
            model = PeftModel.from_pretrained(model, tmp)
        model = model.merge_and_unload()
        print("LoRA merged.")
    model.eval()

    # Companion to grid_thw-on-CPU fix: keep audio_seqlens on CPU so the
    # separate-stream layout (video block → audio block) doesn't produce a
    # CPU+CUDA scalar mix in get_rope_index. Safe in interleaved mode too.
    # See CLAUDE.md "Separate-audio device crash" for full diagnosis.
    _orig_get_rope_index = model.get_rope_index

    def _get_rope_index_cpu_audio(*rope_args, **rope_kwargs):
        rope_args = list(rope_args)
        if len(rope_args) >= 6 and torch.is_tensor(rope_args[5]):
            rope_args[5] = rope_args[5].cpu()
        if torch.is_tensor(rope_kwargs.get("audio_seqlens")):
            rope_kwargs["audio_seqlens"] = rope_kwargs["audio_seqlens"].cpu()
        return _orig_get_rope_index(*rope_args, **rope_kwargs)

    model.get_rope_index = _get_rope_index_cpu_audio

    processor = Qwen2_5OmniProcessor.from_pretrained(base_model_path)
    return model, processor


# ── scoring + output ──────────────────────────────────────────────────────────

def load_exclude_qids(spec):
    """Resolve --exclude-qids. Either an IntentBench-Prime variant name
    ('full' | 'clean' | 'hard'), read from the vendored scorer in
    eval/score_intentbench.py, or a path to a JSON list of qids."""
    if os.path.isfile(spec):
        with open(spec) as f:
            return set(json.load(f))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from score_intentbench import load_exclusion
    return load_exclusion(spec)


def compute_accuracy(results):
    total, correct = 0, 0
    for r in results:
        if r.get("load_error"):
            continue
        pred = extract_answer(r["output"])
        gt = extract_answer(r["solution"])
        correct += reward_fn(pred, gt, r["problem_type"])
        total += 1
    return correct / total if total else 0.0, total


def save_results(results, accuracy, n_total, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"final_acc": {"mean_acc": accuracy, "n_total": n_total}, "results": results},
            f, indent=2, ensure_ascii=False,
        )
    print(f"Saved → {output_path}  (acc={accuracy*100:.2f}%, n={n_total})")


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    if args.modality in ("caption", "both") and not args.caption_json:
        raise SystemExit("--caption-json is required for --modality caption/both")
    if args.hov2_ckpt and args.modality != "video":
        raise SystemExit("--hov2-ckpt is only valid with --modality video")

    if args.video_max_token_num is not None:
        import qwen_omni_utils.v2_5.vision_process as _vp
        _vp.VIDEO_MAX_TOKEN_NUM = args.video_max_token_num
        print(f"VIDEO_MAX_TOKEN_NUM overridden to {args.video_max_token_num}")

    torch.distributed.init_process_group(
        backend="nccl",
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        rank=int(os.getenv("RANK", "0")),
    )
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    model, processor = load_model(args.base_model_path, args.lora_path)

    # Interleaved TMRoPE is the default (our SFT models + base Qwen); pass
    # --separate-audio-stream to reproduce HumanOmniV2's separate-stream protocol.
    audio_in_video = not args.separate_audio_stream

    dataset = IBDirectDataset(
        args.ib_path, args.video_root,
        caption_json=args.caption_json,
        modality=args.modality,
        audio_enabled=not args.no_audio,
        audio_in_video=audio_in_video,
        hov2_ckpt=args.hov2_ckpt,
        fps=args.fps, max_frames=args.max_frames,
    )
    if args.exclude_qids:
        drop = load_exclude_qids(args.exclude_qids)
        before = len(dataset.data)
        dataset.data = [s for s in dataset.data if s.get("qid") not in drop]
        print(f"Excluded {before - len(dataset.data)} qids → {len(dataset.data)} samples ({args.exclude_qids})")
    if args.filter_type:
        types = {t.strip().lower() for t in args.filter_type.split(",")}
        dataset.data = [s for s in dataset.data if s.get("Type", "").lower() in types]
        print(f"Filtered to {len(dataset.data)} samples (Type: {', '.join(sorted(types))})")
    if args.shuffle_seed is not None:
        # Deterministic shuffle so --max-samples yields the SAME fixed random
        # subset for every model (fairness for latency comparison).
        random.Random(args.shuffle_seed).shuffle(dataset.data)
        print(f"Shuffled dataset with seed {args.shuffle_seed}")
    if args.max_samples:
        dataset.data = dataset.data[:args.max_samples]
        print(f"Truncated to {len(dataset.data)} samples")

    os.makedirs(args.results_dir, exist_ok=True)
    if args.hov2_ckpt:
        mode_suffix = "hov2"
    elif args.modality == "video":
        mode_suffix = "direct_answer"   # §4 naming (backward-compatible)
    else:
        mode_suffix = args.modality      # caption / question / both (§5)
    out_path = os.path.join(args.results_dir, f"ib_{args.run_name}_{mode_suffix}.json")
    ckpt_path = f"{out_path}.rank{rank}.ckpt.jsonl"

    max_new_tokens = args.max_new_tokens or (2048 if args.hov2_ckpt else 64)

    results = []
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        if results:
            print(f"Rank {rank}: resuming, skipping {len(results)} samples")
    n_skip = len(results)

    dataloader = DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset), skip=n_skip),
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=lambda x: x,
    )

    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    latency_records = []
    n_timed = 0  # non-error samples processed this rank (for warmup marking)
    wall_start = time.perf_counter()

    for inputs in tqdm(dataloader, desc=f"IB rank{rank}", initial=n_skip, total=n_skip + len(dataloader)):
        sample = inputs[0]  # batch_size=1

        if sample.get("load_error"):
            entry = {
                "qid": sample["qid"], "video": sample["video"],
                "output": "", "solution": sample["solution"],
                "problem_type": sample["problem_type"], "raw": sample["raw"],
                "load_error": True,
            }
            results.append(entry)
            with open(ckpt_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            continue

        try:
            images = list(sample["images"]) if sample["images"] else []
            audios = list(sample["audios"]) if sample["audios"] else []
            videos = list(sample["videos"]) if sample["videos"] else []
            use_aiv = sample.get("use_audio_in_video", False)

            text = processor.apply_chat_template(
                [sample["prompt"]], tokenize=False, add_generation_prompt=True
            )
            model_inputs = processor(
                text=text, audio=audios or None, images=images or None, videos=videos or None,
                return_tensors="pt", padding=True, use_audio_in_video=use_aiv,
            ).to(f"cuda:{local_rank}")
            model_inputs.pop("return_audio", None)

            for k in ("video_grid_thw", "image_grid_thw"):
                if k in model_inputs and model_inputs[k] is not None:
                    model_inputs[k] = model_inputs[k].cpu()

            eos_ids = processor.tokenizer.eos_token_id
            if not isinstance(eos_ids, list):
                eos_ids = [eos_ids]
            im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end_id not in eos_ids:
                eos_ids = eos_ids + [im_end_id]

            input_len = model_inputs.input_ids.size(1)

            if args.latency:
                # Time prefill (time-to-first-token, incl. vision/audio encoders
                # + LM prefill — the cost shared by both models) vs decode (per-
                # token generation — where plain-SFT and reasoning traces diverge).
                timer = _FirstTokenTimer()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    output_ids = model.generate(
                        **model_inputs, use_audio_in_video=use_aiv,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=eos_ids,
                        logits_processor=LogitsProcessorList([timer]),
                    )
                torch.cuda.synchronize()
                t_end = time.perf_counter()

                out_tok = int(output_ids.shape[1] - input_len)
                prefill_s = (timer.t_first - t0) if timer.t_first is not None else None
                total_s = t_end - t0
                decode_s = (t_end - timer.t_first) if timer.t_first is not None else None
                is_warmup = n_timed < args.warmup
                n_timed += 1
                latency_records.append({
                    "qid": sample["qid"], "problem_type": sample["problem_type"],
                    "input_tokens": int(input_len), "output_tokens": out_tok,
                    "prefill_s": prefill_s, "decode_s": decode_s, "total_s": total_s,
                    "warmup": is_warmup, "load_error": False,
                })
            else:
                with torch.inference_mode():
                    output_ids = model.generate(
                        **model_inputs, use_audio_in_video=use_aiv,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=eos_ids,
                    )
            response = processor.decode(
                output_ids[0][model_inputs.input_ids.size(1):],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception as e:
            import traceback
            print(f"[WARN] rank{rank} inference failed: {e}")
            traceback.print_exc()
            response = ""

        entry = {
            "qid": sample["qid"], "video": sample["video"],
            "output": response, "solution": sample["solution"],
            "problem_type": sample["problem_type"], "raw": sample["raw"],
            "load_error": False,
        }
        results.append(entry)
        with open(ckpt_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    wall_clock_s = time.perf_counter() - wall_start

    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])

    all_results = [None] * world_size
    torch.distributed.all_gather_object(all_results, results)

    if args.latency:
        gathered = [None] * world_size
        torch.distributed.all_gather_object(
            gathered, {"records": latency_records, "wall_clock_s": wall_clock_s}
        )

    if rank == 0:
        flat = list(itertools.chain.from_iterable(all_results))
        acc, n = compute_accuracy(flat)
        print(f"IB direct-answer accuracy: {acc*100:.2f}% over {n} samples")
        save_results(flat, acc, n, out_path)
        for r in range(world_size):
            try:
                os.remove(f"{out_path}.rank{r}.ckpt.jsonl")
            except FileNotFoundError:
                pass

        if args.latency:
            all_records = list(itertools.chain.from_iterable(g["records"] for g in gathered))
            # Total wall-clock = slowest rank (ranks run concurrently on separate GPUs).
            wall_total = max(g["wall_clock_s"] for g in gathered)
            config = {
                "run_name": args.run_name,
                "mode": mode_suffix,
                "modality": args.modality,
                "device": torch.cuda.get_device_name(local_rank),
                "world_size": world_size,
                "batch_size": 1,
                "decoding": "greedy (do_sample=False)",
                "max_new_tokens": max_new_tokens,
                "fps": args.fps,
                "max_frames": args.max_frames,
                "audio_in_video": audio_in_video,
                "shuffle_seed": args.shuffle_seed,
                "warmup_per_rank": args.warmup,
            }
            summary = summarize_latency(all_records, wall_total, config)
            lat_path = os.path.join(args.results_dir, f"ib_{args.run_name}_{mode_suffix}_latency.json")
            with open(lat_path, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "per_sample": all_records}, f, indent=2)
            ot, ps, ds = summary["output_tokens"], summary["prefill_s"], summary["decode_s"]
            print(f"\n=== LATENCY ({summary['n_timed']} timed, {summary['n_warmup_skipped']} warmup skipped) ===")
            print(f"  output tokens/answer : mean {ot['mean']:.1f}  median {ot['median']:.0f}  (max {ot['max']})")
            print(f"  prefill  s/question  : mean {ps['mean']:.3f}  median {ps['median']:.3f}")
            print(f"  decode   s/question  : mean {ds['mean']:.3f}  median {ds['median']:.3f}")
            print(f"  total    s/question  : mean {summary['total_s']['mean']:.3f}  median {summary['total_s']['median']:.3f}")
            print(f"  wall-clock total     : {wall_total:.1f}s over {world_size} GPU(s)")
            print(f"Saved → {lat_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--lora-path",        default=None, help="LoRA adapter to merge (omit for base model)")
    parser.add_argument("--ib-path",          required=True, help="IntentBench qa.json")
    parser.add_argument("--video-root",       required=True, help="Directory containing IB videos")
    parser.add_argument("--modality",         choices=["video", "caption", "both", "question"], default="video",
                        help="Information setting: video only (§4, default) / caption only / both / "
                             "question (question+options only, no video and no caption). caption & "
                             "question are text-only; caption/both back the §5 SFT-variants experiment.")
    parser.add_argument("--caption-json",     default=None,
                        help="IB captions JSON (required for --modality caption/both), e.g. eval/ib_captions/asid_7B.json")
    parser.add_argument("--results-dir",      default="eval_results")
    parser.add_argument("--run-name",         default="model", help="Tag for output filenames")
    parser.add_argument("--no-audio",         action="store_true", help="Disable audio (treat all videos as silent)")
    parser.add_argument("--separate-audio-stream", action="store_true", help="Separate-stream audio (HOv2 protocol: explicit audio element + text prefix). Default is interleaved TMRoPE (our SFT models + base Qwen); pass this to reproduce HumanOmniV2's protocol.")
    parser.add_argument("--hov2-ckpt",        action="store_true", help="HOv2 thinker prompt + <answer>-tag templates + 2048-token generation (HOv2 checkpoint only)")
    parser.add_argument("--max-new-tokens",   type=int,   default=None, help="Override generation length (default: 64, or 2048 with --hov2-ckpt)")
    parser.add_argument("--filter-type",      default=None, help="Comma-separated Type values to evaluate")
    parser.add_argument("--fps",              type=float, default=2)
    parser.add_argument("--max-frames",       type=int,   default=32)
    parser.add_argument("--video-max-token-num", type=int, default=None)
    parser.add_argument("--max-samples",      type=int,   default=None, help="Truncate to N samples (smoke testing / latency subset)")
    parser.add_argument("--exclude-qids",     default=None, help="Drop qids before eval: an IntentBench-Prime variant name ('clean' | 'hard'), read from the vendored eval/score_intentbench.py scorer, or a path to a JSON list of qids. Use 'hard' for the IntentBench-hard latency subset.")
    parser.add_argument("--shuffle-seed",     type=int,   default=None, help="Deterministically shuffle the dataset before --max-samples so every model sees the same fixed random subset (latency runs)")
    parser.add_argument("--latency",          action="store_true", help="Measure per-question prefill/decode wall-clock + output-token counts. Writes ib_{run}_{mode}_latency.json. Run on 1 GPU, batch=1, for clean numbers.")
    parser.add_argument("--warmup",           type=int,   default=3, help="Latency mode: number of leading samples per rank excluded from summary stats (GPU/cuDNN warmup)")
    args = parser.parse_args()
    main(args)
