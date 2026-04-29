"""
Inspect Power-SMC / baseline result JSONL files.

Prints headline accuracy, finish-reason and token distributions, prediction
vs ground-truth letter distribution, and sample completions. Handles single
files (full report) or multiple files (side-by-side comparison table).

Usage (from repo root):

    # Full report on one file
    python scripts/inspect_results.py results/pmc_vqa_qwen_power_smc_cot_v100.jsonl

    # Show 3 sample completions (1 correct, 1 wrong, 1 None-prediction)
    python scripts/inspect_results.py results/pmc_vqa_qwen_power_smc_cot_v100.jsonl --samples 3

    # Side-by-side comparison across models
    python scripts/inspect_results.py \\
        results/pmc_vqa_qwen_power_smc_cot_v100.jsonl \\
        results/pmc_vqa_llava_med_power_smc_cot_v100.jsonl \\
        results/pmc_vqa_medgemma_power_smc_cot_v100.jsonl

    # Just the headline numbers (compact mode)
    python scripts/inspect_results.py results/*.jsonl --compact

    # Smoke-test mode: 1-question file, dump everything
    python scripts/inspect_results.py results/smoke_medgemma_eos_fix.jsonl --smoke
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


# ─── Loading ─────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> tuple[list[dict], dict | None]:
    """Return (per-question records, summary record-or-None)."""
    questions: list[dict] = []
    summary: dict | None = None
    with open(path, "r", encoding="utf-8") as f:
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
    return questions, summary


def get_method_block(rec: dict) -> tuple[str, dict]:
    """Return (method_name, method_dict). Supports power_smc and baseline."""
    methods = rec.get("methods", {})
    for name in ("power_smc", "baseline"):
        if name in methods:
            return name, methods[name]
    if methods:
        name = next(iter(methods))
        return name, methods[name]
    return "unknown", {}


# ─── Stats ──────────────────────────────────────────────────────────────────

def compute_stats(questions: list[dict]) -> dict[str, Any]:
    """Aggregate per-sample stats across all questions."""
    correct = total = 0
    pred_dist: Counter = Counter()
    gt_dist: Counter = Counter()
    finish_counts: Counter = Counter()
    n_tokens_total = 0
    n_tokens_count = 0
    method_seen = "unknown"
    q_types: Counter = Counter()

    for r in questions:
        method_seen, m = get_method_block(r)
        correct += m.get("n_correct", 0)
        total   += m.get("n_rollouts", 0)
        gt_dist[r.get("ground_truth")] += 1
        q_types[r.get("question_type", "unknown")] += 1
        for s in m.get("samples", []):
            pred_dist[s.get("predicted")] += 1
            finish_counts[s.get("finish_reason", "unknown")] += 1
            n_tokens_total += s.get("n_tokens", 0)
            n_tokens_count += 1

    return {
        "method":          method_seen,
        "n_questions":     len(questions),
        "n_correct":       correct,
        "n_total":         total,
        "accuracy":        correct / total if total else 0.0,
        "pred_dist":       pred_dist,
        "gt_dist":         gt_dist,
        "finish_counts":   finish_counts,
        "avg_tokens":      n_tokens_total / n_tokens_count if n_tokens_count else 0.0,
        "n_tokens_count":  n_tokens_count,
        "q_types":         q_types,
    }


# ─── Reporting ──────────────────────────────────────────────────────────────

def _fmt_pct(n: int, total: int) -> str:
    return f"{(100 * n / total):.1f}%" if total else "—"


def _fmt_dist(d: Counter, total: int | None = None) -> str:
    """Pretty-print a Counter as 'A: 30 (30.0%), B: 29 (29.0%), ...'."""
    if total is None:
        total = sum(d.values())
    items = sorted(d.items(), key=lambda kv: (kv[0] is None, str(kv[0])))
    return ", ".join(
        f"{k!r}: {v} ({_fmt_pct(v, total)})" for k, v in items
    )


def print_full_report(path: Path, samples_to_show: int = 0) -> dict:
    questions, summary = load_jsonl(path)
    stats = compute_stats(questions)

    print(f"\n{'=' * 78}")
    print(f"  {path}")
    print(f"{'=' * 78}")
    print(f"  method            : {stats['method']}")
    print(f"  questions written : {stats['n_questions']}")
    print(f"  question types    : {dict(stats['q_types'])}")
    print(f"  accuracy          : {stats['accuracy']:.4f}  "
          f"({stats['n_correct']}/{stats['n_total']})")
    print(f"  avg tokens/sample : {stats['avg_tokens']:.1f}")
    print(f"  finish reasons    : {dict(stats['finish_counts'])}")

    n_total = sum(stats["pred_dist"].values())
    n_null = stats["pred_dist"].get(None, 0)
    if n_null:
        print(f"  null predictions  : {n_null}/{n_total}  "
              f"({_fmt_pct(n_null, n_total)})  ← parser failed to extract answer")

    print(f"  GT letter dist    : {_fmt_dist(stats['gt_dist'])}")
    print(f"  predicted dist    : {_fmt_dist(stats['pred_dist'])}")

    if summary and "summary" in summary:
        print(f"\n  --- summary record from JSONL ---")
        for method_name, s in summary["summary"].items():
            for k, v in s.items():
                if isinstance(v, float):
                    print(f"    {method_name}.{k:<22}: {v:.4f}")
                else:
                    print(f"    {method_name}.{k:<22}: {v}")

    # Optional: show sample completions
    if samples_to_show > 0:
        _print_sample_completions(questions, samples_to_show)

    return stats


def _print_sample_completions(questions: list[dict], n_each: int) -> None:
    """Show up to n_each correct, n_each wrong, and any None-prediction examples."""
    print(f"\n  --- sample completions (up to {n_each} of each type) ---")

    correct_qs = [q for q in questions
                  if any(s["correct"] for s in get_method_block(q)[1].get("samples", []))]
    wrong_qs = [q for q in questions
                if not any(s["correct"] for s in get_method_block(q)[1].get("samples", []))]
    null_qs = [q for q in questions
               if any(s.get("predicted") is None
                      for s in get_method_block(q)[1].get("samples", []))]

    for tag, pool in (("CORRECT", correct_qs), ("WRONG", wrong_qs), ("NULL-PRED", null_qs)):
        for q in pool[:n_each]:
            _, m = get_method_block(q)
            s = m["samples"][0]
            print(f"\n  [{tag}] idx={q['idx']}  GT={q['ground_truth']!r}  "
                  f"pred={s['predicted']!r}  correct={s['correct']}  "
                  f"tokens={s['n_tokens']}  finish={s['finish_reason']}")
            comp = (s.get("completion") or "").strip()
            if len(comp) > 600:
                print(f"    {comp[:300]} ... [{len(comp)-500} chars] ... {comp[-200:]}")
            else:
                print(f"    {comp}")


# ─── Side-by-side ───────────────────────────────────────────────────────────

def print_comparison_table(paths: list[Path]) -> None:
    rows = []
    for p in paths:
        questions, _ = load_jsonl(p)
        s = compute_stats(questions)
        rows.append((p.name, s))

    if not rows:
        return

    name_w = max(len(name) for name, _ in rows)

    print(f"\n{'=' * 78}")
    print("  Side-by-side comparison")
    print(f"{'=' * 78}\n")

    header = (
        f"  {'file':<{name_w}}  "
        f"{'questions':>9}  "
        f"{'accuracy':>9}  "
        f"{'samples':>8}  "
        f"{'avg toks':>9}  "
        f"{'null pred':>10}  "
        f"{'finish=stop':>12}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name, s in rows:
        n_total = s["n_total"] or 1
        n_null = s["pred_dist"].get(None, 0)
        n_stop = s["finish_counts"].get("stop", 0)
        print(
            f"  {name:<{name_w}}  "
            f"{s['n_questions']:>9}  "
            f"{s['accuracy']:>9.4f}  "
            f"{s['n_total']:>8}  "
            f"{s['avg_tokens']:>9.1f}  "
            f"{_fmt_pct(n_null, n_total):>10}  "
            f"{_fmt_pct(n_stop, n_total):>12}"
        )
    print()


# ─── Smoke mode ─────────────────────────────────────────────────────────────

def print_smoke(path: Path) -> None:
    """Compact view for a 1-question smoke result."""
    questions, _ = load_jsonl(path)
    if not questions:
        print(f"  (no question records in {path})")
        return

    print(f"\n{'=' * 78}")
    print(f"  SMOKE: {path}")
    print(f"{'=' * 78}")

    for q in questions:
        _, m = get_method_block(q)
        for s in m.get("samples", []):
            print(f"\n  idx={q['idx']}  rollout={s['sample_idx']}")
            print(f"    GT             : {q['ground_truth']!r}")
            print(f"    Predicted      : {s['predicted']!r}")
            print(f"    Correct        : {s['correct']}")
            print(f"    Tokens         : {s['n_tokens']}")
            print(f"    Finish reason  : {s['finish_reason']}")
            comp = (s.get("completion") or "").strip()
            print(f"    Completion ({len(comp)} chars):")
            print(f"      first 300 : {comp[:300]!r}")
            if len(comp) > 300:
                print(f"      last 200  : {comp[-200:]!r}")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Inspect Power-SMC / baseline result JSONL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("paths", nargs="+", type=Path,
                   help="JSONL result file(s)")
    p.add_argument("--samples", type=int, default=0,
                   help="Per file, show this many CORRECT, WRONG, and NULL-PRED "
                        "sample completions (default: 0 = none)")
    p.add_argument("--compact", action="store_true",
                   help="Skip the full per-file report; show only the comparison "
                        "table (most useful with multiple files)")
    p.add_argument("--smoke", action="store_true",
                   help="Compact 1-question smoke view (predicted, correct, "
                        "tokens, finish_reason, completion preview)")
    args = p.parse_args()

    missing = [str(p) for p in args.paths if not p.exists()]
    if missing:
        print("ERROR: file(s) not found:")
        for m in missing:
            print(f"  {m}")
        return

    if args.smoke:
        for path in args.paths:
            print_smoke(path)
        return

    if not args.compact:
        for path in args.paths:
            print_full_report(path, samples_to_show=args.samples)

    if len(args.paths) > 1:
        print_comparison_table(args.paths)


if __name__ == "__main__":
    main()
