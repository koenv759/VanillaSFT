#!/usr/bin/env python3
"""Score an existing IntentBench eval file against the IntentBench-Prime variants.

This is a **vendored, self-contained copy** of the scorer from IntentBench-Prime
(https://github.com/koenv759/IntentBench-Prime), dropped into this repo so
VanillaSFT needs no external package to reproduce the numbers. The only change
from upstream is that the variants manifest is loaded from the sibling
``ib_data/`` folder by relative path instead of via ``importlib.resources``. If
the upstream exclusion lists ever change, refresh the three JSONs in
``eval/ib_data/``.

IntentBench-Prime is a removal-only cleaning of IntentBench: it drops broken
questions (Tier A) and Social-IQ questions answerable from text alone (Tier B).
Because it only removes questions, you can re-score any result file produced on
the *original* IntentBench without re-running the model.

Usage
-----
    # Full / Clean / Hard accuracy side by side (uses the bundled variants manifest):
    python eval/score_intentbench.py results.json

    # Report the two tiny who/which + when categories separately:
    python eval/score_intentbench.py results.json --no-merge-small

    # As a library (from the eval/ dir): the public helpers are load_exclusion and accuracy
    from score_intentbench import accuracy, load_exclusion
    acc, n = accuracy(results, "hard")

Expected result-file schema
---------------------------
A JSON object with a "results" list; each item has:
    output      str   the model's raw generation (an <answer>...</answer> is parsed if present)
    solution    str   the ground-truth answer (same parsing)
    raw         dict  with keys: qid, problem_type ('multiple choice'|'judge'|'emer_ov_mc'), Type
Items may carry "load_error": true to be skipped.

Depends only on the Python standard library.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Bundled manifest, shipped alongside this file in eval/ib_data/.
DEFAULT_VARIANTS = Path(__file__).resolve().parent / "ib_data" / "intentbench_variants.json"


def load_variants(manifest_path=None):
    """Return the parsed variants manifest (bundled by default)."""
    path = Path(manifest_path) if manifest_path else DEFAULT_VARIANTS
    with open(path) as f:
        return json.load(f)


def load_exclusion(variant, manifest_path=None):
    """Return the set of qids excluded by a variant ('full' | 'clean' | 'hard').

    'full' excludes nothing. 'clean' drops the 192 broken questions; 'hard' also
    drops the 616 text-answerable ones (790 unique). Accepts the legacy 'easy'
    spelling of 'clean'.
    """
    variant = variant.strip().lower()
    variants = load_variants(manifest_path)["variants"]
    if variant not in variants and variant == "clean" and "easy" in variants:
        variant = "easy"
    if variant not in variants:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(variants)}")
    return set(variants[variant].get("exclude", []))


def extract_answer(text):
    """Pull the final answer out of a raw generation (or ground-truth) string.

    Preferred form is an explicit ``<answer>...</answer>`` span, which reasoning
    models emit. If that tag is absent we fall back to scraping standalone A-F
    letters: this covers plain-letter outputs and multi-label EMER answers such
    as "A, C, D" that arrive without tags. The same function is applied to both
    the model output and the ground truth so they are parsed identically.
    """
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: all standalone A-F letters (handles multi-label EMER without tags)
    letters = re.findall(r'\b([A-F])\b', text)
    if letters:
        return ','.join(dict.fromkeys(letters))  # deduplicate, preserve order
    return ''


def _norm_judge(ans):
    # A=Yes, B=No. Accept either the letter (base Qwen outputs A/B) or the word
    # (HumanOmniV2 outputs Yes/No) on BOTH the model output and the GT.
    a = ans.strip().lower()
    if a in ('yes', 'a'):
        return 'a'
    if a in ('no', 'b'):
        return 'b'
    return a


def judge(output_ans, gt_ans):
    # Binary yes/no question (deception): correct iff the normalized answers match.
    return 1.0 if _norm_judge(output_ans) == _norm_judge(gt_ans) else 0.0


def emer_ov_mc(reference, hypothesis):
    """Soft F1 between two comma-separated multi-label answers (EMER emotion).

    EMER questions can have several correct emotions, so instead of exact match
    we score the token-set overlap as an F1: precision against the reference
    labels, recall against the predicted labels. Returns 0 when there is no
    overlap (or either side is empty).
    """
    list_a = reference.split(',')
    list_b = hypothesis.split(',')
    true_positive = len(set(list_a) & set(list_b))
    precision = true_positive / len(list_a) if list_a else 0
    recall = true_positive / len(list_b) if list_b else 0
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    return 0


def reward_fn(output_ans, gt_ans, question_type):
    """Dispatch to the scoring rule for a question's ``problem_type``.

    Each rule returns a reward in [0, 1] (1.0 = fully correct):
      - 'multiple choice' : exact letter match (Social-IQ why/how/what/who/when).
      - 'emer_ov_mc'      : soft label-overlap F1 (EMER emotion, multi-label).
      - 'judge'           : yes/no match (deception).
    An unknown type scores 0.0 rather than raising, so a stray question can't
    crash a whole run.
    """
    if question_type == 'multiple choice':
        return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
    elif question_type == 'emer_ov_mc':
        return emer_ov_mc(output_ans, gt_ans)
    elif question_type == 'judge':
        return judge(output_ans, gt_ans)
    return 0.0


# The two tiny Social-IQ categories are folded into 'other' when merge_small is on
# (default). who/which=25 and when=14 are too small to report on their own.
MERGE_SMALL = {'who/which', 'when'}


def score(results, excluded, merge_small=True):
    """Score one result list, dropping ``excluded`` qids and unusable samples.

    Walks every result once and returns four things:
      - ``by_type``   : {category -> list of per-question rewards}
      - ``overall``   : flat list of every scored reward
      - ``n_load_err``: samples skipped because the eval marked ``load_error``
      - ``n_excl``    : samples skipped because their qid is in ``excluded``

    Passing an empty ``excluded`` set scores the Full benchmark; passing a
    variant's exclusion list scores Clean or Hard. Accuracy is later computed as
    the mean of a reward list, so keeping the raw lists lets the caller report
    both the accuracy and the N behind it.
    """
    by_type = defaultdict(list)
    overall = []
    n_load_err = 0
    n_excl = 0
    for r in results:
        # Sample the eval itself failed to produce (e.g. video failed to load).
        if r.get('load_error'):
            n_load_err += 1
            continue
        # Question removed by the active IntentBench-Prime variant.
        if r['raw'].get('qid') in excluded:
            n_excl += 1
            continue
        final_ans = extract_answer(r['output'])
        gt_ans = extract_answer(r['solution'])
        reward = reward_fn(final_ans, gt_ans, r['raw']['problem_type'])
        cat = r['raw'].get('Type', 'unknown')
        if merge_small and cat in MERGE_SMALL:
            cat = 'other'
        by_type[cat].append(reward)
        overall.append(reward)
    return by_type, overall, n_load_err, n_excl


def accuracy(results, variant="full", merge_small=True, manifest_path=None):
    """Convenience: overall accuracy (%) and N for one variant of a result list.

    ``results`` is the list under a result file's ``"results"`` key. Returns
    ``(acc_percent, n_scored)``. Use ``main_variants`` for the full per-category
    Full/Clean/Hard table.
    """
    excluded = set() if variant.strip().lower() == "full" else load_exclusion(variant, manifest_path)
    _, overall, _, _ = score(results, excluded, merge_small)
    return (sum(overall) / len(overall) * 100 if overall else 0.0), len(overall)


def cat_order(merge_small):
    # Fixed left-to-right ordering for the per-category rows so tables from
    # different files line up. Categories not listed here are appended, sorted,
    # by the callers. When merge_small is on, the two tiny categories are hidden
    # (their questions are counted under 'other' instead).
    order = ["why", "how", "what", "when", "who/which", "other", "emotion", "deception"]
    if merge_small:
        order = [c for c in order if c not in MERGE_SMALL]
    return order


def main_variants(results, manifest_path, merge_small):
    """Print every variant the manifest defines (Full / Clean / Hard) side by side.

    This is the default view. The manifest maps each variant name to the qids it
    excludes, so we score the same result list once per variant and lay the
    per-category accuracies out in columns.
    """
    manifest = json.load(open(manifest_path))
    variants = manifest['variants']
    names = list(variants.keys())  # manifest order (full, clean, hard)
    excl = {n: set(variants[n].get('exclude', [])) for n in names}

    # Header: for each variant show how many of its excluded qids are actually
    # present in THIS file, so a partial or mismatched result file is obvious.
    all_qids = {r['raw'].get('qid') for r in results if not r.get('load_error')}
    print(f"Manifest: {manifest.get('name', manifest_path)}  ({manifest.get('built', '')})")
    for n in names:
        present = len(excl[n] & all_qids)
        print(f"  {n:<5} exclude {len(excl[n]):>4}  ({present} present here)")
    print()

    # Score the file once per variant. n_load_err is the same every pass (it
    # doesn't depend on the exclusion set), so we just keep the last one.
    by_type, overall, n_load_err = {}, {}, None
    for n in names:
        bt, ov, nle, _ = score(results, excl[n], merge_small)
        by_type[n], overall[n] = bt, ov
        n_load_err = nle
    if n_load_err:
        print(f"(excluded {n_load_err} samples with load_error)\n")

    order = cat_order(merge_small)
    cats = order + sorted(
        c for c in set().union(*(set(by_type[n]) for n in names)) if c not in order
    )

    def acc(lst):
        return f"{sum(lst)/len(lst)*100:.2f}" if lst else "  -  "

    hdr = f"{'Type':<12}"
    for n in names:
        hdr += f" {n[:8].capitalize():>8} {'N':>6}  "
    print(hdr)
    print('-' * len(hdr))
    for cat in cats:
        if not any(by_type[n].get(cat) for n in names):
            continue
        row = f"{cat:<12}"
        for n in names:
            s = by_type[n].get(cat, [])
            row += f" {acc(s):>8} {len(s):>6}  "
        print(row)
    print('-' * len(hdr))
    row = f"{'Overall':<12}"
    for n in names:
        row += f" {acc(overall[n]):>8} {len(overall[n]):>6}  "
    print(row)


def main(args):
    # One behavior: score the result file against the variants manifest and print
    # Full / Clean / Hard side by side. --variants only exists so you can point at
    # a different manifest; --no-merge-small only changes the category breakdown.
    with open(args.results_path) as f:
        data = json.load(f)
    main_variants(data['results'], args.variants, args.merge_small)


def _cli():
    parser = argparse.ArgumentParser(
        description="Score an existing IntentBench result file against IntentBench-Prime. "
                    "Prints Full / Clean / Hard accuracy side by side, by category.")
    parser.add_argument('results_path', type=str,
                        help="Path to the IntentBench eval results JSON file.")
    parser.add_argument('--variants', type=str, default=str(DEFAULT_VARIANTS),
                        help="Path to the variants manifest that defines Full/Clean/Hard. "
                             "Defaults to the bundled ib_data/intentbench_variants.json; only "
                             "override it to score against a custom manifest.")
    parser.add_argument('--merge-small', action=argparse.BooleanOptionalAction, default=True,
                        help="Fold the two tiny who/which + when categories into 'other' "
                             "(default on; use --no-merge-small to report them separately).")
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    _cli()
