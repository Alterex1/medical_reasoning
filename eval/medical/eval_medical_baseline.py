"""
Medical VQA Eval — Baseline (greedy / temperature sampling)

Baseline comparison for Power-SMC on medical VQA. Uses standard
model.generate() with greedy or temperature sampling instead of
particle-based inference.

Same prompts, grading, and output format as eval_medical_smc.py
so results are directly comparable.

Usage (from repo root):
    # Greedy baseline (deterministic, 1 rollout)
    python -m eval.medical.eval_medical_baseline \
        --dataset       data/vqa_rad/VQA_RAD_test.json \
        --model         Qwen/Qwen2.5-VL-7B-Instruct \
        --output        results/medical_baseline_greedy.jsonl \
        --dtype         bfloat16 \
        --temperature   0.0 \
        --max_new_tokens 512 \
        --max_examples  5

    # LLaVA-Med baseline
    python -m eval.medical.eval_medical_baseline \
        --model microsoft/llava-med-v1.5-mistral-7b ...

    # Resume a crashed run
    python -m eval.medical.eval_medical_baseline ... --resume
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from eval.medical.medical_grader import (
    grade_medical_answer,
    is_closed_ended,
    parse_medical_answer,
    parse_mcq_answer,
)
from eval.medical.model_adapters import create_adapter, get_prompt
from eval.medical.eval_medical_smc import load_completed, pass_at_k


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_eval(args):
    # ── check dataset exists ──────────────────────────────────────────────────
    if not Path(args.dataset).exists():
        print(f"Dataset not found: {args.dataset}")
        print("Run: python data/download_medqa.py")
        sys.exit(1)

    # ── dataset ───────────────────────────────────────────────────────────────
    with open(args.dataset) as f:
        dataset = json.load(f)
    start = args.start_index
    end   = start + args.max_examples if args.max_examples is not None else len(dataset)
    dataset = dataset[start:end]
    print(f"Loaded {len(dataset)} questions from {args.dataset} (idx {start}–{end-1})")

    # ── resume ────────────────────────────────────────────────────────────────
    completed_indices = set()
    total_correct     = 0
    total_samples     = 0
    out_mode          = "w"

    if args.resume:
        completed_indices, total_correct, total_samples = load_completed(args.output)
        if completed_indices:
            out_mode = "a"
            print(
                f"[resume] {len(completed_indices)} questions already done — "
                f"skipping ({total_correct}/{total_samples} correct so far)."
            )
        else:
            print("[resume] No existing output found — starting fresh.")

    # ── model ─────────────────────────────────────────────────────────────────
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model} ...")
    adapter = create_adapter(
        args.model, dtype_map[args.dtype], device,
        model_type=args.model_type,
    )
    model = adapter.model
    print(f"Model loaded on {device}")

    # ── decoding config ───────────────────────────────────────────────────────
    do_sample = args.temperature > 0.0
    method_name = f"baseline_temp{args.temperature}"
    print(
        f"\nBaseline config: temperature={args.temperature}  "
        f"do_sample={do_sample}  n_rollouts={args.n_rollouts}\n"
    )

    # ── output ────────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    closed_correct = 0
    closed_total   = 0
    open_correct   = 0
    open_total     = 0

    with open(args.output, out_mode, encoding="utf-8") as fout:
        for idx, data in enumerate(tqdm(dataset, desc="Questions")):
            abs_idx = start + idx

            if abs_idx in completed_indices:
                continue

            gt         = data["answer"]
            question   = data["question"]
            image_path = data["image"]
            q_type     = data.get("question_type")
            if q_type not in ("mcq", "closed", "open"):
                q_type = "closed" if is_closed_ended(gt) else "open"

            # ── build VLM inputs ─────────────────────────────────────────────
            prompt_text = get_prompt(question, cot=args.cot, question_type=q_type)
            gen_inputs = adapter.prepare_generate_inputs(
                question, image_path, prompt_text
            )
            prompt_len = gen_inputs.pop("prompt_len")

            samples_out = []

            for r in tqdm(range(args.n_rollouts), leave=False,
                          desc=f"  idx={abs_idx} rollouts"):

                with torch.inference_mode():
                    generated_ids = model.generate(
                        **gen_inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=do_sample,
                        temperature=args.temperature if do_sample else None,
                    )

                gen_tokens = generated_ids[0, prompt_len:]
                n_tokens   = int(gen_tokens.shape[0])
                completion = adapter.tokenizer.decode(
                    gen_tokens.cpu().tolist(), skip_special_tokens=True
                )

                # Check finish reason — match against the model's full stop
                # token set, not just tokenizer.eos_token_id (which is a
                # single int and misses e.g. Gemma's <end_of_turn>).
                gen_cfg_eos = getattr(model.generation_config, "eos_token_id", None)
                if gen_cfg_eos is None:
                    gen_cfg_eos = adapter.tokenizer.eos_token_id
                eos_ids_set = set(
                    [int(e) for e in gen_cfg_eos]
                    if isinstance(gen_cfg_eos, (list, tuple))
                    else [int(gen_cfg_eos)]
                )
                if n_tokens > 0 and int(gen_tokens[-1].item()) in eos_ids_set:
                    finish_reason = "stop"
                else:
                    finish_reason = "length"

                if q_type == "mcq":
                    predicted = parse_mcq_answer(completion)
                else:
                    predicted = parse_medical_answer(completion, ground_truth=gt)
                correct = grade_medical_answer(predicted, gt, question_type=q_type)

                samples_out.append({
                    "sample_idx":    r,
                    "completion":    completion,
                    "predicted":     predicted,
                    "correct":       correct,
                    "finish_reason": finish_reason,
                    "n_tokens":      n_tokens,
                })

                del generated_ids
                torch.cuda.empty_cache()

            n = len(samples_out)
            c = sum(s["correct"] for s in samples_out)
            total_correct += c
            total_samples += n

            if q_type == "closed":
                closed_correct += c
                closed_total   += n
            else:
                open_correct += c
                open_total   += n

            fout.write(json.dumps({
                "idx":          abs_idx,
                "question":     question,
                "image":        image_path,
                "ground_truth": gt,
                "question_type": q_type,
                "methods": {
                    "baseline": {
                        "samples":    samples_out,
                        "n_correct":  c,
                        "n_rollouts": n,
                        "pass_at_k":  {str(k): pass_at_k(n, c, k) for k in range(1, n + 1)},
                    }
                },
            }) + "\n")
            fout.flush()
            os.fsync(fout.fileno())

        # ── summary ───────────────────────────────────────────────────────────
        acc = total_correct / max(total_samples, 1)
        closed_acc = closed_correct / max(closed_total, 1)
        open_acc   = open_correct / max(open_total, 1)

        print(f"\n[BASELINE]  Overall   = {acc:.4f}  ({total_correct}/{total_samples})")
        print(f"[BASELINE]  Closed    = {closed_acc:.4f}  ({closed_correct}/{closed_total})")
        print(f"[BASELINE]  Open      = {open_acc:.4f}  ({open_correct}/{open_total})")

        fout.write(json.dumps({
            "_type":       "summary",
            "model":       args.model,
            "dataset":     args.dataset,
            "n_questions":  len(dataset),
            "start_index": args.start_index,
            "n_rollouts":  args.n_rollouts,
            "baseline_config": {
                "temperature": args.temperature,
                "do_sample":   do_sample,
            },
            "summary": {
                "baseline": {
                    "overall_accuracy":  acc,
                    "closed_accuracy":   closed_acc,
                    "open_accuracy":     open_acc,
                    "n_correct":         total_correct,
                    "n_total":           total_samples,
                    "closed_correct":    closed_correct,
                    "closed_total":      closed_total,
                    "open_correct":      open_correct,
                    "open_total":        open_total,
                }
            },
        }) + "\n")
        fout.flush()

    print(f"Results written to {args.output}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Medical VQA baseline eval (greedy / temperature)")

    p.add_argument("--dataset",           type=str,   default="data/vqa_rad/VQA_RAD_test.json")
    p.add_argument("--model",             type=str,   default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--model-type",       type=str,   default=None,
                   choices=["qwen", "llava", "medgemma"],
                   help="VLM type (auto-detected from model name if omitted).")
    p.add_argument("--output",            type=str,   default="results/medical_baseline.jsonl")
    p.add_argument("--n_rollouts",        type=int,   default=1,
                   help="Number of independent completions per question. "
                        "Use 1 for greedy, >1 for temperature sampling.")
    p.add_argument("--max_new_tokens",    type=int,   default=512)
    p.add_argument("--start_index",       type=int,   default=0)
    p.add_argument("--max_examples",      type=int,   default=None)
    p.add_argument("--dtype",             type=str,   default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--temperature",       type=float, default=0.0,
                   help="Sampling temperature. 0.0 = greedy (deterministic).")
    p.add_argument("--cot",    action="store_true",  default=True)
    p.add_argument("--no-cot", dest="cot",           action="store_false")
    p.add_argument("--resume", action="store_true")

    args = p.parse_args()
    run_eval(args)
