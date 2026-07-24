"""
Eval script for Daily-Omni and WorldSense benchmarks.
Adapted from HumanOmniV2's eval_humanomniv2.py.

Prompt modes:
  (default)     Plain system prompt, no format suffix. For base Qwen and our SFT models
                (both output raw letters). extract_answer() tries <answer> tags first,
                falls back to response.strip() — handles both formats transparently.
  --plain-sft   Alias for default; explicitly marks our plain-answer SFT models.
  --hov2-ckpt   HOv2 thinker system prompt + <answer> tag suffix. Only for HOv2 checkpoint.

Audio protocol:
  (default)     interleaved audio (Qwen-native TMRoPE): audio extracted from the
                video, no text prefix. For our SFT models and base Qwen.
  --separate-audio-stream
                separate audio stream + "Here is a video..." text prefix — pass to
                reproduce HumanOmniV2's own evaluation protocol.

Usage (Jean-Zay, N GPUs):
  python -m torch.distributed.launch --use_env --nproc_per_node N eval/eval_benchmarks.py \
      --model-path <path> --dataset daily --gt-path <qa.json> --video-root <dir> \
      --file-name <run_name> [--lora-path <path>] [--plain-sft | --hov2-ckpt]
"""

import os
import json
import re
import time
import itertools
import argparse
import shutil
import tempfile

import torch
from peft import PeftModel
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
from safetensors.torch import load_file, save_file
import av


# ── helpers ──────────────────────────────────────────────────────────────────

def check_if_video_has_audio(video_path):
    try:
        container = av.open(video_path)
        return any(s.type == "audio" for s in container.streams)
    except Exception:
        return False


def extract_answer(text):
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    return m.group(1).strip() if m else ""


def normalize_number(s):
    try:
        return float(s.replace(',', ''))
    except Exception:
        return None


def reward_fn(pred, gt, problem_type):
    try:
        if problem_type == "multiple choice":
            return 1.0 if pred.strip() == gt.strip() else 0.0
        elif problem_type in ("numerical", "regression"):
            pn, gn = normalize_number(pred), normalize_number(gt)
            if pn is None or gn is None:
                return 0.0
            return 1.0 if round(pn, 2) == round(gn, 2) else 0.0
        elif problem_type == "emer_ov_mc":
            la, lb = gt.split(","), pred.split(",")
            tp = len(set(la) & set(lb))
            p = tp / len(la) if la else 0
            r = tp / len(lb) if lb else 0
            return 2 * p * r / (p + r) if p + r > 0 else 0.0
        return 0.0
    except Exception:
        return 0.0


# ── system prompts ────────────────────────────────────────────────────────────

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

PLAIN_SYSTEM_PROMPT = "You are a helpful assistant."
PLAIN_TYPE_TEMPLATE = " Answer with only the option letter (e.g., A, B, C, D)."

# Appended to prompts for thinker mode; empty for plain-SFT (raw letter output).
# NOTE: only the "multiple choice" entry is ever used — both Daily-Omni and
# WorldSense are pure MCQ and the dataset classes hardcode problem_type to
# "multiple choice". The other keys are vestigial (carried from HOv2's mixed-type
# eval) and never accessed.
THINKER_TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "emer_ov_mc":      " Please provide only the single or multiple option letter (e.g., A for single option or A,E for multi option, etc.) within the <answer> </answer> tags.",
    "numerical":       " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "regression":      " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "free-form":       " Please provide your text answer within the <answer> </answer> tags.",
    "OCR":             " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "judge":           " Please answer Yes or No within the <answer> </answer> tags.",
}

# ── dataset classes ───────────────────────────────────────────────────────────

def _load_media_with_retry(message, use_aiv, retries=4):
    """process_mm_info with retries. Video decode can throw transient I/O errors
    under multi-rank lustre contention (decord EOF-retry cap, then a torchvision
    fallback BlockingIOError [Errno 11]) on files that are otherwise fine. Retry
    with backoff before giving up; the caller turns a final failure into a
    load_error sentinel so one bad read can't kill the whole distributed run.
    """
    last = None
    for attempt in range(retries):
        try:
            return process_mm_info(message, use_audio_in_video=use_aiv)
        except Exception as e:
            last = e
            time.sleep(0.5 * (attempt + 1))
    raise last


def _rotate_options(candidates, answer, shift):
    """Cyclically rotate option *content* across the fixed letter slots, by `shift`
    positions, and return (new_candidates, new_answer).

    Each candidate looks like "A. some text". The letter labels (A, B, C, ...) stay
    in place; the texts move so that the text originally at index i appears at index
    (i + shift) % n. The correct-answer letter is remapped accordingly. Used by the
    positional-bias diagnostic (env WORLD_OPTION_SHIFT). shift=0 is a no-op.
    """
    if not shift:
        return candidates, answer
    parsed = []
    for c in candidates:
        m = re.match(r"\s*([A-Z])\s*[.)]\s*(.*)", c, re.DOTALL)
        if not m:
            return candidates, answer  # unexpected format; leave untouched
        parsed.append((m.group(1), m.group(2)))
    letters = [p[0] for p in parsed]
    texts = [p[1] for p in parsed]
    n = len(parsed)
    shift = shift % n
    # new text at position i is the old text from position (i - shift)
    new_texts = [texts[(i - shift) % n] for i in range(n)]
    new_candidates = [f"{letters[i]}. {new_texts[i]}" for i in range(n)]
    # the answer's text moves from old index a to new index (a + shift)
    try:
        a = letters.index(answer.strip())
    except ValueError:
        return new_candidates, answer
    new_answer = letters[(a + shift) % n]
    return new_candidates, new_answer


def _build_message(video_path, text_prompt, system_prompt, audio_in_video=False):
    """Build the message list for a single video+text input.

    audio_in_video=False (HOv2 protocol): audio attached as a separate stream
        element + "Here is a video, with the audio..." text prefix.
    audio_in_video=True (interleaved, Qwen-native): video element only — audio is
        extracted from the video by process_mm_info and time-aligned via TMRoPE.
        No prefix (matches our interleaved training data: just <video>question).

    Returns (message, use_audio_in_video) — the flag is per-sample: True only
    when interleaved mode is on AND the video actually has an audio track.
    """
    has_audio = check_if_video_has_audio(video_path)
    _fps = float(os.getenv("EVAL_FPS", "2"))
    _max_frames = int(os.getenv("EVAL_MAX_FRAMES", "32"))
    user_content = [
        {"type": "video", "video": video_path, "fps": _fps, "max_frames": _max_frames},
    ]
    if audio_in_video:
        user_content.append({"type": "text", "text": text_prompt})
        use_aiv = has_audio
    elif has_audio:
        user_content.append({"type": "audio", "audio": video_path})
        user_content.append({"type": "text", "text": f"Here is a video, with the audio from the video.\n{text_prompt}"})
        use_aiv = False
    else:
        user_content.append({"type": "text", "text": text_prompt})
        use_aiv = False

    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user",   "content": user_content},
    ], use_aiv


class DailyOmniDataset(Dataset):
    """
    Raw Daily-Omni QA format.
    QA file: flat list with fields Question, Choice (list "A. text"), Answer (raw letter),
             video_id, Type, video_duration.
    Video path: {video_root}/Videos/{video_id}/{video_id}_video.mp4
    """
    def __init__(self, qa_path, video_root, processor, plain_sft, audio_in_video=False, hov2_ckpt=False):
        with open(qa_path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.video_root = video_root
        self.processor = processor
        self.plain_sft = plain_sft
        self.hov2_ckpt = hov2_ckpt
        self.audio_in_video = audio_in_video
        self.system_prompt = THINKER_SYSTEM_PROMPT if hov2_ckpt else PLAIN_SYSTEM_PROMPT

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        question = item["Question"] + " Options:\n" + "\n".join(item["Choice"])
        suffix = THINKER_TYPE_TEMPLATE["multiple choice"] if self.hov2_ckpt else PLAIN_TYPE_TEMPLATE
        text_prompt = question + suffix

        video_id = item["video_id"]
        video_path = os.path.join(self.video_root, "Videos", video_id, f"{video_id}_video.mp4")
        message, use_aiv = _build_message(video_path, text_prompt, self.system_prompt, self.audio_in_video)
        try:
            audios, images, videos = _load_media_with_retry(message, use_aiv)
        except Exception as e:
            print(f"[WARN] media load failed for {video_path}: {e}")
            return {"load_error": True, "prompt": message, "use_audio_in_video": use_aiv,
                    "images": [], "audios": [], "videos": [],
                    "gt_answer": item["Answer"], "problem_type": "multiple choice",
                    "raw": {**item, "video_path": video_path}}

        return {
            "images": images,
            "audios": audios,
            "videos": videos,
            "prompt": message,
            "use_audio_in_video": use_aiv,
            "gt_answer": item["Answer"],          # raw letter — no tag extraction needed
            "problem_type": "multiple choice",
            "raw": {**item, "video_path": video_path},
        }


class WorldSenseDataset(Dataset):
    """
    Raw WorldSense QA format.
    QA file: dict keyed by video_id; each entry has task0, task1… with
             question, answer (raw letter), candidates (list "A. text").
    Video path: {video_root}/{video_id}.mp4
    """
    def __init__(self, qa_path, video_root, processor, plain_sft, audio_in_video=False, hov2_ckpt=False):
        raw = json.load(open(qa_path, encoding="utf-8"))
        self.data = []
        for vid, entry in raw.items():
            video_meta = {k: v for k, v in entry.items() if not k.startswith("task")}
            for key, task in entry.items():
                if not key.startswith("task"):
                    continue
                self.data.append({
                    **video_meta,
                    "task_key":    key,
                    "task_domain": task.get("task_domain"),
                    "task_type":   task.get("task_type"),
                    "question":    task["question"],
                    "candidates":  task["candidates"],
                    "answer":      task["answer"],
                })
        self.video_root = video_root
        self.processor = processor
        self.plain_sft = plain_sft
        self.hov2_ckpt = hov2_ckpt
        self.audio_in_video = audio_in_video
        self.system_prompt = THINKER_SYSTEM_PROMPT if hov2_ckpt else PLAIN_SYSTEM_PROMPT

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        shift = int(os.getenv("WORLD_OPTION_SHIFT", "0"))
        candidates, answer = _rotate_options(item["candidates"], item["answer"], shift)
        question = item["question"] + " Options:\n" + "\n".join(candidates)
        suffix = THINKER_TYPE_TEMPLATE["multiple choice"] if self.hov2_ckpt else PLAIN_TYPE_TEMPLATE
        text_prompt = question + suffix

        video_path = os.path.join(self.video_root, f"{item['video_id']}.mp4")
        message, use_aiv = _build_message(video_path, text_prompt, self.system_prompt, self.audio_in_video)
        raw = {**item, "video_path": video_path,
               "orig_answer": item["answer"], "shifted_answer": answer,
               "option_shift": shift}
        try:
            audios, images, videos = _load_media_with_retry(message, use_aiv)
        except Exception as e:
            print(f"[WARN] media load failed for {video_path}: {e}")
            return {"load_error": True, "prompt": message, "use_audio_in_video": use_aiv,
                    "images": [], "audios": [], "videos": [],
                    "gt_answer": answer, "problem_type": "multiple choice", "raw": raw}

        return {
            "images": images,
            "audios": audios,
            "videos": videos,
            "prompt": message,
            "use_audio_in_video": use_aiv,
            "gt_answer": answer,                   # raw letter (shifted if WORLD_OPTION_SHIFT)
            "problem_type": "multiple choice",
            "raw": raw,
        }


DATASET_CLASSES = {
    "daily": DailyOmniDataset,
    "world": WorldSenseDataset,
}


# ── distributed sampler ───────────────────────────────────────────────────────

class InferenceSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, size, skip=0):
        self._size = int(size)
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = [i for i in range(size) if i % self._world_size == self._rank]
        self._skip = skip

    def __iter__(self):
        yield from self._local_indices[self._skip:]

    def __len__(self):
        return len(self._local_indices) - self._skip


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_path, lora_path):
    """Load base model + optionally merge a LoRA adapter.

    HOv2 checkpoints store adapter keys as base_model.model.thinker.*  and
    target_modules as a regex containing 'thinker\\.'.  Plain PeftModel.from_pretrained
    fails on these; we patch both in a tempdir before loading, matching eval_ib.py.
    """
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    # Some Omni checkpoints ship a FULL qwen2_5_omni config.json (thinker_config/
    # talker_config/token2wav_config) even when the weights are thinker-only. Feeding
    # that full config to the Thinker's config_class narrows the audio encoder wrongly
    # (d_model 1280 built as 3584) -> size-mismatch crash. Parse via AutoConfig and
    # hand from_pretrained the correctly-built thinker sub-config. No-op for a bare
    # qwen2_5_omni_thinker config (no .thinker_config attr) — HOv2 / our SFT checkpoints.
    _cfg = AutoConfig.from_pretrained(model_path)
    thinker_cfg = getattr(_cfg, "thinker_config", None)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
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
            cfg["base_model_name_or_path"] = model_path
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

    # Workaround, companion to the grid_thw-on-CPU fix below: with grid_thw on
    # CPU, get_rope_index's running offset `st` becomes a CPU scalar after a
    # video block, while audio_seqlens (from feature_attention_mask) is CUDA.
    # In the separate-stream layout (video block then audio block) the audio
    # branch then adds CPU + CUDA scalars → device error. Moving audio_seqlens
    # to CPU keeps all position arithmetic on CPU; the function returns
    # positions via .to(position_ids.device) so the output is unaffected.
    # NOTE: eval_ib.py only escapes this by operand-ordering luck in the
    # interleaved branch — it MUST get this same wrapper when it grows a
    # --separate-audio mode (CLAUDE.md todo 16).
    _orig_get_rope_index = model.get_rope_index

    def _get_rope_index_cpu_audio(*rope_args, **rope_kwargs):
        rope_args = list(rope_args)
        if len(rope_args) >= 6 and torch.is_tensor(rope_args[5]):
            rope_args[5] = rope_args[5].cpu()
        if torch.is_tensor(rope_kwargs.get("audio_seqlens")):
            rope_kwargs["audio_seqlens"] = rope_kwargs["audio_seqlens"].cpu()
        return _orig_get_rope_index(*rope_args, **rope_kwargs)

    model.get_rope_index = _get_rope_index_cpu_audio

    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    return model, processor


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        rank=int(os.getenv("RANK", "0")),
    )
    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    model, processor = load_model(args.model_path, args.lora_path)

    # Interleaved TMRoPE is the default (our SFT models + base Qwen); pass
    # --separate-audio-stream to reproduce HumanOmniV2's separate-stream protocol.
    audio_in_video = not args.separate_audio_stream

    DatasetClass = DATASET_CLASSES[args.dataset]
    dataset = DatasetClass(args.gt_path, args.video_root, processor, args.plain_sft,
                           audio_in_video=audio_in_video, hov2_ckpt=args.hov2_ckpt)
    if args.max_samples:
        dataset.data = dataset.data[:args.max_samples]
        print(f"Truncated dataset to {len(dataset.data)} samples (smoke test)")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.dataset}_{args.file_name}.json")
    ckpt_path = f"{out_path}.rank{rank}.ckpt.jsonl"

    # Resume: reload any entries already written by a previous run on this rank.
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
            print(f"Rank {rank}: resuming, skipping {len(results)} already-processed samples")

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

    for inputs in tqdm(dataloader, desc=f"Evaluating {args.dataset} rank{rank}",
                       initial=n_skip, total=n_skip + len(dataloader)):
        sample = inputs[0]  # batch_size=1

        if sample.get("load_error"):
            entry = {
                "gt_answer":    sample["gt_answer"],
                "output":       "",
                "prompt":       sample["prompt"],
                "problem_type": sample["problem_type"],
                "raw":          sample["raw"],
                "load_error":   True,
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
            ).to(f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")
            model_inputs.pop("return_audio", None)

            # Keep grid tensors on CPU — CUDA int64 prod() kernel broken on this platform.
            for k in ("video_grid_thw", "image_grid_thw"):
                if k in model_inputs and model_inputs[k] is not None:
                    model_inputs[k] = model_inputs[k].cpu()

            eos_ids = processor.tokenizer.eos_token_id
            if not isinstance(eos_ids, list):
                eos_ids = [eos_ids]
            im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if im_end_id not in eos_ids:
                eos_ids = eos_ids + [im_end_id]

            with torch.inference_mode():
                output_ids = model.generate(
                    **model_inputs, use_audio_in_video=use_aiv,
                    max_new_tokens=args.max_new_tokens,
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
            "gt_answer":    sample["gt_answer"],
            "output":       response,
            "prompt":       sample["prompt"],
            "problem_type": sample["problem_type"],
            "raw":          sample["raw"],
        }
        results.append(entry)
        with open(ckpt_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    torch.distributed.barrier()

    gts     = [e["gt_answer"]    for e in results]
    rets    = [e["output"]       for e in results]
    sources = [{"prompt": e["prompt"], "gt_answer": e["gt_answer"],
                "problem_type": e["problem_type"], "raw": e["raw"]} for e in results]

    merged_gts, merged_rets, merged_sources = [None]*world_size, [None]*world_size, [None]*world_size
    torch.distributed.all_gather_object(merged_gts,     gts)
    torch.distributed.all_gather_object(merged_rets,    rets)
    torch.distributed.all_gather_object(merged_sources, sources)

    if rank != 0:
        return

    all_gts     = list(itertools.chain.from_iterable(merged_gts))
    all_rets    = list(itertools.chain.from_iterable(merged_rets))
    all_sources = list(itertools.chain.from_iterable(merged_sources))

    final_output = []
    reward_sum = 0.0
    for gt, response, sample in zip(all_gts, all_rets, all_sources):
        if args.plain_sft:
            pred = response.strip()
        else:
            pred = extract_answer(response)
            if not pred:
                pred = response.strip()

        reward = reward_fn(pred, gt, sample["problem_type"])
        reward_sum += reward
        final_output.append({**sample, "output": response, "prediction": pred, "reward": reward})

    mean_acc = reward_sum / len(final_output) if final_output else 0.0
    print(f"{args.dataset} accuracy: {mean_acc:.4f}  ({len(final_output)} items)")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": final_output, "mean_acc": mean_acc}, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")

    for r in range(world_size):
        try:
            os.remove(f"{out_path}.rank{r}.ckpt.jsonl")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path",     required=True)
    parser.add_argument("--dataset",        required=True, choices=["daily", "world"])
    parser.add_argument("--gt-path",        required=True, help="Path to QA JSON file")
    parser.add_argument("--video-root",     required=True, help="Root directory for videos")
    parser.add_argument("--file-name",      default="eval",  help="Tag for output filename")
    parser.add_argument("--output-dir",     default="eval_results")
    parser.add_argument("--lora-path",      default=None,   help="LoRA adapter path to merge")
    parser.add_argument("--plain-sft",      action="store_true",
                        help="Alias for default; marks our plain-answer SFT models explicitly")
    parser.add_argument("--hov2-ckpt",      action="store_true",
                        help="HOv2 checkpoint mode: thinker system prompt + <answer> tag suffix")
    parser.add_argument("--separate-audio-stream", action="store_true",
                        help="Separate-stream audio (HOv2 protocol: separate audio stream + "
                             "text prefix). Default is interleaved TMRoPE (our SFT models + "
                             "base Qwen); pass this to reproduce HumanOmniV2's protocol.")
    parser.add_argument("--max-new-tokens", type=int, default=10,
                        help="Max tokens to generate (10 for plain-SFT, 2048 for thinker)")
    parser.add_argument("--max-samples",   type=int, default=None,
                        help="Truncate dataset to N samples (smoke testing only)")
    args = parser.parse_args()
    main(args)
