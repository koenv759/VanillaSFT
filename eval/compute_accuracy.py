"""
Compute per-category accuracy for Daily-Omni, WorldSense, and IntentBench.
Reads all result JSONs from eval_results/, outputs a printed table and
eval_results/results_summary.xlsx with one sheet per benchmark.

Usage:
    python eval/compute_all_accuracy.py [--results-dir eval_results] [--output eval_results/results_summary.xlsx]
"""
import json
import re
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

DAILY_CATS = [
    "AV Event Alignment",
    "Comparative",
    "Context understanding",
    "Event Sequence",
    "Inference",
    "Reasoning",
]
DAILY_DURATION_COLS = ["30s", "60s"]
DAILY_COLS = DAILY_CATS + DAILY_DURATION_COLS + ["Avg"]

WORLD_CATS = [
    "Tech & Science",
    "Culture & Politics",
    "Daily Life",
    "Film & TV",
    "Performance",
    "Games",
    "Sports",
    "Music",
]
WORLD_COLS = WORLD_CATS + ["Avg"]

IB_CATS = ["why", "how", "what", "when", "who/which", "other", "emotion", "deception"]
IB_COLS = IB_CATS + ["Overall"]


# ---------------------------------------------------------------------------
# IB scoring (mirrors compute_ib_accuracy.py)
# ---------------------------------------------------------------------------

def _extract_answer(text):
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    letters = re.findall(r'\b([A-F])\b', text)
    if letters:
        return ','.join(dict.fromkeys(letters))
    return ''


def _emer_f1(ref, hyp):
    a, b = ref.split(','), hyp.split(',')
    tp = len(set(a) & set(b))
    prec = tp / len(a) if a else 0
    rec = tp / len(b) if b else 0
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0


def _ib_reward(output_ans, gt_ans, problem_type):
    if problem_type == 'multiple choice':
        return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
    elif problem_type == 'emer_ov_mc':
        return _emer_f1(gt_ans, output_ans)
    elif problem_type == 'judge':
        def _norm(x):
            x = x.strip().lower()
            if 'yes' in x:
                return 'a'
            if 'no' in x:
                return 'b'
            return x
        return 1.0 if _norm(output_ans) == _norm(gt_ans) else 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Per-benchmark loaders
# ---------------------------------------------------------------------------

def load_daily(path):
    d = json.load(open(path))
    by_type = defaultdict(list)
    by_dur = defaultdict(list)
    for r in d['results']:
        if r.get('load_error'):
            continue
        score = r['reward']
        cat = r['raw']['Type']
        dur = r['raw']['video_duration']
        by_type[cat].append(score)
        by_dur[dur].append(score)

    all_scores = [r['reward'] for r in d['results'] if not r.get('load_error')]
    row, counts = {}, {}
    for c in DAILY_CATS:
        scores = by_type.get(c, [])
        row[c] = (sum(scores) / len(scores) * 100) if scores else None
        counts[c] = len(scores)
    for dur in DAILY_DURATION_COLS:
        scores = by_dur.get(dur, [])
        row[dur] = (sum(scores) / len(scores) * 100) if scores else None
        counts[dur] = len(scores)
    row['Avg'] = (sum(all_scores) / len(all_scores) * 100) if all_scores else None
    counts['Avg'] = len(all_scores)
    return row, counts


def load_world(path):
    d = json.load(open(path))
    by_domain = defaultdict(list)
    for r in d['results']:
        if r.get('load_error'):
            continue
        by_domain[r['raw']['domain']].append(r['reward'])

    all_scores = [r['reward'] for r in d['results'] if not r.get('load_error')]
    row, counts = {}, {}
    for c in WORLD_CATS:
        scores = by_domain.get(c, [])
        row[c] = (sum(scores) / len(scores) * 100) if scores else None
        counts[c] = len(scores)
    row['Avg'] = (sum(all_scores) / len(all_scores) * 100) if all_scores else None
    counts['Avg'] = len(all_scores)
    return row, counts


def load_ib(path):
    d = json.load(open(path))
    by_type = defaultdict(list)
    for r in d['results']:
        if r.get('load_error'):
            continue
        output_ans = _extract_answer(r['output'])
        gt_ans = _extract_answer(r['solution'])
        problem_type = r['problem_type']
        score = _ib_reward(output_ans, gt_ans, problem_type)
        cat = r['raw'].get('Type', 'unknown')
        by_type[cat].append(score)

    all_scores = [s for scores in by_type.values() for s in scores]
    row, counts = {}, {}
    for c in IB_CATS:
        scores = by_type.get(c, [])
        row[c] = (sum(scores) / len(scores) * 100) if scores else None
        counts[c] = len(scores)
    row['Overall'] = (sum(all_scores) / len(all_scores) * 100) if all_scores else None
    counts['Overall'] = len(all_scores)
    return row, counts


# ---------------------------------------------------------------------------
# Collect all results
# ---------------------------------------------------------------------------

def _prepend_n_row(rows, cols, load_fn):
    """Build a DataFrame with an N (samples) row prepended above the run rows."""
    acc_rows, count_rows = {}, {}
    for run, (acc, counts) in rows.items():
        acc_rows[run] = acc
        count_rows[run] = counts

    df = pd.DataFrame(acc_rows, index=cols).T

    # N is the same for all runs (same dataset); use first available
    if count_rows:
        n_counts = next(iter(count_rows.values()))
        n_row = pd.DataFrame({col: [n_counts.get(col, None)] for col in cols}, index=['N (samples)'])
        df = pd.concat([n_row, df])

    return df


def collect(results_dir):
    results_dir = Path(results_dir)
    daily_rows, world_rows, ib_rows = {}, {}, {}

    for f in sorted(results_dir.glob('*.json')):
        name = f.stem  # e.g. daily_coldstart_e2
        if '.rank' in name or '.ckpt' in name:
            continue

        if name.startswith('daily_'):
            run = name[len('daily_'):]
            try:
                daily_rows[run] = load_daily(f)
            except Exception as e:
                print(f"[WARN] {f.name}: {e}")

        elif name.startswith('world_'):
            run = name[len('world_'):]
            try:
                world_rows[run] = load_world(f)
            except Exception as e:
                print(f"[WARN] {f.name}: {e}")

        elif name.startswith('ib_'):
            # strip leading ib_ and trailing _direct_answer or _hov2
            run = re.sub(r'_(direct_answer|hov2)$', '', name[len('ib_'):])
            if name.endswith('_hov2'):
                run += '_hov2'
            try:
                ib_rows[run] = load_ib(f)
            except Exception as e:
                print(f"[WARN] {f.name}: {e}")

    return (
        _prepend_n_row(daily_rows, DAILY_COLS, load_daily),
        _prepend_n_row(world_rows, WORLD_COLS, load_world),
        _prepend_n_row(ib_rows, IB_COLS, load_ib),
    )


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_table(title, df):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print('='*80)
    if df.empty:
        print("  (no results found)")
        return
    def _fmt_row(row):
        if row.name == 'N (samples)':
            return row.map(lambda x: str(int(x)) if x is not None and not pd.isna(x) else "—")
        return row.map(lambda x: f"{x:.2f}" if x is not None and not pd.isna(x) else "—")
    fmt = df.apply(_fmt_row, axis=1)
    print(fmt.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='eval_results')
    parser.add_argument('--output', default=None,
                        help="Path for the .xlsx summary (default: <results-dir>/results_summary.xlsx)")
    args = parser.parse_args()

    df_daily, df_world, df_ib = collect(args.results_dir)

    print_table("Daily-Omni (accuracy %)", df_daily)
    print_table("WorldSense (accuracy %)", df_world)
    print_table("IntentBench (accuracy %)", df_ib)

    # Write Excel next to the results by default, so pointing --results-dir at any
    # existing folder (e.g. results/vanilla_lora_e1) just works.
    out_path = Path(args.output) if args.output else Path(args.results_dir) / "results_summary.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        df_daily.round(2).to_excel(writer, sheet_name='Daily-Omni')
        df_world.round(2).to_excel(writer, sheet_name='WorldSense')
        df_ib.round(2).to_excel(writer, sheet_name='IntentBench')

    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
