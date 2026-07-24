"""
Build the Vanilla-SFT training set from HumanOmniV2's rewrite JSONs.

HumanOmniV2 releases its training data ("long CoT" rewrites) as three JSON files
(social_iq_v2_rewrite.json, emer_rewrite.json, Video-R1_rewrite.json). This script
reads them and writes a plain-answer ms-swift SFT JSONL: the reasoning traces
(<context>...</context><think>...</think>) are discarded and the training target
is the raw answer field (single letter for MCQ, raw text for numerical/free-form).

The result is `matched_alldata.jsonl` — the exact §4 training set (30,217 samples),
byte-for-byte reproducing HumanOmniV2's data selection, minus the traces. A copy
with repo-relative video paths ships in `data/`; run this to regenerate it with
absolute paths pointing at your local video root.

Get the rewrite JSONs + videos from HumanOmniV2's release:
  - Rewrite JSONs / curated data: https://huggingface.co/datasets/PhilipC/IntentTrain
  - Component video sources (the only three the 30,217 examples reference):
    Video-R1 (github.com/tulerfeng/Video-R1), Social-IQ 2.0, MER2024/EMER.

Usage:
  python data_prep/convert.py \
    --config-dir /path/to/rewrite_jsons \
    --data-root  /path/to/videos \
    --output-dir data
"""

import argparse
import json
import pathlib

# The three rewrite JSONs, in the order HumanOmniV2 unions them.
SOURCES = [
    "emer_rewrite.json",
    "social_iq_v2_rewrite.json",
    "Video-R1_rewrite.json",
]

EXPECTED_TOTAL = 30217

# Prompt format matches HOv2's eval_humanomniv2.py exactly (without their
# TYPE_TEMPLATE suffix, which instructs <answer> tags we don't use).


def build_example(rec, data_root):
    problem = rec["problem"]
    options = rec.get("options") or []
    answer = rec["answer"]
    path = rec["path"]
    is_video = rec.get("data_type", "video") == "video"

    if options:
        user_text = problem + " Options:\n" + "\n".join(options)
    else:
        user_text = problem

    media_tag = "<video>" if is_video else "<image>"
    media_key = "videos" if is_video else "images"

    return {
        "messages": [
            {"role": "user",      "content": f"{media_tag}{user_text}"},
            {"role": "assistant", "content": answer},
        ],
        media_key: [str(data_root / path)],
    }


def write_jsonl(records, out_path, data_root):
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(build_example(rec, data_root), ensure_ascii=False) + "\n")
    print(f"  {out_path.name}: {len(records)} examples")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root",  required=True,
                        help="Absolute path to the videos/images root")
    parser.add_argument("--config-dir", required=True,
                        help="Directory containing the three HumanOmniV2 rewrite JSONs")
    parser.add_argument("--output-dir", default="data",
                        help="Directory for the output JSONL")
    args = parser.parse_args()

    data_root  = pathlib.Path(args.data_root)
    config_dir = pathlib.Path(args.config_dir)
    out_dir    = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for filename in SOURCES:
        recs = json.load(open(config_dir / filename))
        print(f"{filename}: {len(recs)} records")
        all_records.extend(recs)

    assert len(all_records) == EXPECTED_TOTAL, \
        f"Expected {EXPECTED_TOTAL} records, got {len(all_records)}"

    print(f"\nWriting JSONL to {out_dir}/")
    write_jsonl(all_records, out_dir / "matched_alldata.jsonl", data_root)
    print("Done.")


if __name__ == "__main__":
    main()
