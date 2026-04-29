"""
Re-grade existing eval JSONLs with the current parser/grader.

Useful when the grading logic improves (e.g. fuzzy MCQ fallback added) and
you want to apply it to results you already collected without re-running
inference. Reads the original JSONL, re-runs ``parse_mcq_answer`` and
``grade_medical_answer`` on each sample's stored completion, and writes a
new JSONL with updated ``predicted`` / ``correct`` / ``n_correct`` /
``pass_at_k`` / summary.

The original input is never modified.

Usage (from repo root):

    # Single file
    python scripts/regrade_results.py \\
        results/pmc_vqa_llava_med_power_smc_cot_v100.jsonl \\
        --out results/pmc_vqa_llava_med_power_smc_cot_v100_regraded.jsonl

    # Multiple files at once (--out is auto-derived as ``<stem>_regraded.jsonl``)
    python scripts/regrade_results.py results/pmc_vqa_*v100*.jsonl

    # In-place style: overwrite the input (NOT recommended unless you have backups)
    python scripts/regrade_results.py file.jsonl --in-place

After regrading, inspect with:

    python scripts/inspect_results.py results/foo_regraded.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Make the repo root importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.medical.medical_grader import (
    grade_medical_answer,
    is_closed_ended,
    parse_medical_answer,
    parse_mcq_answer,
)


# Match PMC-VQA-style choice lines like:
#   A) X-ray
#   B) MRI
#   C) CT
#   D) Ultrasound
_CHOICE_LINE_RE = re.compile(r"^\s*([A-D])\s*[):.\-]\s*(.+?)\s*$")


def extract_choices_from_question(question: str) -> dict | None:
    """
    Recover the choices dict from a PMC-VQA-style question string when the
    eval JSONL didn't save it explicitly.

    Returns ``{"A": "...", "B": "...", "C": "...", "D": "..."}`` if all four
    choices are found, else None.
    """
    if not question:
        return None
    choices: dict = {}
    for line in question.splitlines():
        m = _CHOICE_LINE_RE.match(line)
        if m:
            letter = m.group(1).upper()
            text = m.group(2).strip()
            if letter in "ABCD" and text and letter not in choices:
                choices[letter] = text
    if len(choices) == 4:
        return choices
    return None


def _pass_at_k(n: int, c: int, k: int):
    if n < k:
        return None
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    prob = 1.0
    for i in range(k):
        prob *= (n - c - i) / (n - i)
    return 1.0 - prob


def regrade_record(rec: dict) -> dict:
    """Return a new record with re-graded predictions/correctness."""
    if rec.get("_type") == "summary":
        return rec  # summary records are rebuilt from scratch by caller

    gt = rec["ground_truth"]
    q_type = rec.get("question_type")
    if q_type not in ("mcq", "closed", "open"):
        q_type = "closed" if is_closed_ended(gt) else "open"
    # Older eval JSONLs didn't save `choices` — parse them back out of the
    # question text (PMC-VQA embeds them as "A) ... B) ..." lines).
    choices = rec.get("choices") or extract_choices_from_question(rec.get("question", ""))

    methods = {}
    for method_name, m in rec.get("methods", {}).items():
        new_samples = []
        n_correct = 0
        for s in m.get("samples", []):
            completion = s.get("completion") or ""
            if q_type == "mcq":
                predicted = parse_mcq_answer(completion, choices=choices)
            else:
                predicted = parse_medical_answer(completion, ground_truth=gt)
            correct = grade_medical_answer(predicted, gt, question_type=q_type)
            n_correct += int(bool(correct))
            new_samples.append({**s, "predicted": predicted, "correct": correct})
        n_rollouts = len(new_samples)
        methods[method_name] = {
            **m,
            "samples":    new_samples,
            "n_correct":  n_correct,
            "n_rollouts": n_rollouts,
            "pass_at_k":  {str(k): _pass_at_k(n_rollouts, n_correct, k)
                           for k in range(1, n_rollouts + 1)},
        }
    return {**rec, "methods": methods, "question_type": q_type}


def rebuild_summary(records: list[dict], original_summary: dict | None) -> dict:
    """Rebuild a summary record from regraded per-question records."""
    closed_correct = closed_total = 0
    open_correct = open_total = 0
    n_correct = n_total = 0
    method_seen = "power_smc"

    for r in records:
        for method_name, m in r.get("methods", {}).items():
            method_seen = method_name
            n_correct += m.get("n_correct", 0)
            n_total   += m.get("n_rollouts", 0)
            qt = r.get("question_type", "open")
            if qt == "closed":
                closed_correct += m.get("n_correct", 0)
                closed_total   += m.get("n_rollouts", 0)
            else:
                open_correct += m.get("n_correct", 0)
                open_total   += m.get("n_rollouts", 0)

    summary_payload = {
        "overall_accuracy": n_correct / max(n_total, 1),
        "closed_accuracy":  closed_correct / max(closed_total, 1) if closed_total else 0.0,
        "open_accuracy":    open_correct   / max(open_total, 1)   if open_total   else 0.0,
        "n_correct":        n_correct,
        "n_total":          n_total,
        "closed_correct":   closed_correct,
        "closed_total":     closed_total,
        "open_correct":     open_correct,
        "open_total":       open_total,
    }

    base = original_summary if isinstance(original_summary, dict) else {"_type": "summary"}
    return {
        **base,
        "_type":   "summary",
        "summary": {method_seen: summary_payload},
        "regraded": True,
    }


def regrade_file(in_path: Path, out_path: Path) -> tuple[int, int, int, int]:
    """
    Returns (n_questions, before_correct, after_correct, total_samples).
    """
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    questions: list[dict] = []
    summary: dict | None = None
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("_type") == "summary":
                summary = rec
            else:
                questions.append(rec)

    # Stats before
    before_correct = sum(
        m.get("n_correct", 0)
        for q in questions
        for m in q.get("methods", {}).values()
    )
    total_samples = sum(
        m.get("n_rollouts", 0)
        for q in questions
        for m in q.get("methods", {}).values()
    )

    # Regrade
    new_questions = [regrade_record(q) for q in questions]
    new_summary = rebuild_summary(new_questions, summary)

    after_correct = sum(
        m.get("n_correct", 0)
        for q in new_questions
        for m in q.get("methods", {}).values()
    )

    # Write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for q in new_questions:
            f.write(json.dumps(q) + "\n")
        f.write(json.dumps(new_summary) + "\n")

    return len(new_questions), before_correct, after_correct, total_samples


def main() -> None:
    p = argparse.ArgumentParser(description="Re-grade eval JSONLs with the current parser")
    p.add_argument("paths", nargs="+", type=Path, help="Result JSONL(s) to regrade")
    p.add_argument("--out", type=Path, default=None,
                   help="Output path. Required when regrading a single file unless "
                        "--in-place is set. Ignored when multiple inputs are given "
                        "(outputs auto-named ``<stem>_regraded.jsonl``).")
    p.add_argument("--in-place", action="store_true",
                   help="Overwrite the input file. Use with care.")
    args = p.parse_args()

    if len(args.paths) == 1 and args.out is not None and not args.in_place:
        out_paths = [args.out]
    elif args.in_place:
        out_paths = list(args.paths)
    else:
        out_paths = [
            p.with_name(p.stem + "_regraded" + p.suffix) for p in args.paths
        ]

    print(f"{'file':<60}  {'n_q':>5}  {'before':>10}  {'after':>10}  {'delta':>6}")
    print("-" * 100)
    for in_path, out_path in zip(args.paths, out_paths):
        try:
            n_q, before, after, total = regrade_file(in_path, out_path)
        except FileNotFoundError:
            print(f"{str(in_path):<60}  MISSING")
            continue
        delta = after - before
        sign = "+" if delta >= 0 else ""
        print(
            f"{str(in_path.name):<60}  {n_q:>5}  "
            f"{before}/{total:<6}  {after}/{total:<6}  {sign}{delta:>5}  "
            f"-> {out_path}"
        )


if __name__ == "__main__":
    main()
