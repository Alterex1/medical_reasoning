"""
Medical VQA Eval — Power-SMC sampler with Vision-Language Models

Evaluates Power-SMC on medical visual question answering datasets
(VQA-RAD, SLAKE, etc.) using multimodal models like Qwen2.5-VL,
LLaVA-Med, Med-Gemma, etc.

The core SMC algorithm is reused from power_smc.py — only the evaluation
harness, prompting, and grading are adapted for the medical domain.
Model-specific logic is handled by adapters in model_adapters.py.

Usage:
    # Download dataset first (from repo root)
    python data/download_medqa.py

    # Run evaluation (from repo root)
    python -m eval.medical.eval_medical_smc \
        --dataset       data/vqa_rad/VQA_RAD_test.json \
        --model         Qwen/Qwen2.5-VL-7B-Instruct \
        --output        results/medical_power_smc.jsonl \
        --dtype         bfloat16 \
        --alpha         4.0 \
        --n_particles   64 \
        --n_rollouts    32 \
        --prompt_batch_size 4 \
        --max_new_tokens 256

    # LLaVA-Med (auto-detected from model name)
    python -m eval.medical.eval_medical_smc \
        --model microsoft/llava-med-v1.5-mistral-7b ...

    # Resume a crashed run
    python -m eval.medical.eval_medical_smc ... --resume
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
from core.power_smc import PowerSMC, normalize_weights


def pass_at_k(n: int, c: int, k: int):
    if n < k:     return None
    if c == 0:    return 0.0
    if n - c < k: return 1.0
    prob = 1.0
    for i in range(k):
        prob *= (n - c - i) / (n - i)
    return 1.0 - prob


# ─── Resume helpers ───────────────────────────────────────────────────────────

def load_completed(output_path: str) -> tuple[set, int, int]:
    """
    Read an existing JSONL output and return:
      completed_indices, total_correct, total_samples
    """
    completed = set()
    total_correct = 0
    total_samples = 0
    path = Path(output_path)
    if not path.exists():
        return completed, total_correct, total_samples
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
                continue
            idx = rec.get("idx")
            if idx is None:
                continue
            completed.add(idx)
            methods = rec.get("methods", {})
            # Support both power_smc and baseline method keys
            method_data = methods.get("power_smc") or methods.get("baseline") or {}
            total_correct += method_data.get("n_correct",  0)
            total_samples += method_data.get("n_rollouts", 0)
    return completed, total_correct, total_samples


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_eval(args):
    # ── check dataset exists ──────────────────────────────────────────────────
    if not Path(args.dataset).exists():
        print(f"Dataset not found: {args.dataset}")
        print("Run: python download_medqa.py")
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

    # ── Power-SMC sampler ─────────────────────────────────────────────────────
    sampler = PowerSMC(
        model           = model,
        tokenizer       = adapter.tokenizer,
        alpha           = args.alpha,
        n_particles     = args.n_particles,
        kappa           = args.kappa,
        resample_method = args.resample,
    )
    M = args.prompt_batch_size
    print(
        f"\nPower-SMC config: α={args.alpha}  N={args.n_particles}  "
        f"κ={args.kappa}  resample={args.resample}  "
        f"n_rollouts={args.n_rollouts}  prompt_batch_size={M}\n"
    )

    # ── output ────────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Track closed vs open-ended accuracy separately
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
            # Prefer explicit question_type from the dataset (e.g. PMC-VQA
            # sets "mcq"); fall back to closed/open inference for VQA-RAD.
            q_type     = data.get("question_type")
            if q_type not in ("mcq", "closed", "open"):
                q_type = "closed" if is_closed_ended(gt) else "open"

            # ── build VLM inputs ─────────────────────────────────────────────
            prompt_text = get_prompt(question, cot=args.cot, question_type=q_type)
            vlm_inputs = adapter.prepare_inputs(question, image_path, prompt_text)

            input_ids      = vlm_inputs["input_ids"]
            attn_mask      = vlm_inputs["attention_mask"]
            prompt_len     = vlm_inputs["prompt_len"]
            prefill_kwargs = vlm_inputs["prefill_kwargs"]

            samples_out = []
            eos_id = adapter.tokenizer.eos_token_id

            for chunk_start in tqdm(range(0, args.n_rollouts, M), leave=False,
                                    desc=f"  idx={abs_idx} chunks"):
                actual_M = min(M, args.n_rollouts - chunk_start)

                if actual_M > 1:
                    outputs = sampler.generate_batch(
                        [input_ids] * actual_M,
                        [attn_mask] * actual_M,
                        max_new_tokens=args.max_new_tokens,
                        prefill_kwargs=prefill_kwargs,
                    )
                else:
                    outputs = [sampler.generate(
                        input_ids, attn_mask,
                        max_new_tokens=args.max_new_tokens,
                        prefill_kwargs=prefill_kwargs,
                    )]

                for k, out in enumerate(outputs):
                    r = chunk_start + k

                    chosen_i   = out.chosen_idx
                    gen_tokens = out.sequences[0, prompt_len:]

                    # Trim at first EOS
                    eos_pos = (gen_tokens == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_pos) > 0:
                        gen_tokens    = gen_tokens[:int(eos_pos[0].item()) + 1]
                        finish_reason = "stop"
                    else:
                        finish_reason = "length"

                    n_tokens        = int(gen_tokens.shape[0])
                    completion      = adapter.tokenizer.decode(
                        gen_tokens.cpu().tolist(), skip_special_tokens=True
                    )
                    sum_logprob     = out.chosen_sum_logprob
                    mean_logprob    = sum_logprob / max(n_tokens, 1)
                    smc_log_weight  = float(out.log_weights[chosen_i].item())
                    smc_norm_weight = float(normalize_weights(out.log_weights)[chosen_i].item())

                    if q_type == "mcq":
                        predicted = parse_mcq_answer(completion)
                    else:
                        predicted = parse_medical_answer(completion, ground_truth=gt)
                    correct = grade_medical_answer(predicted, gt, question_type=q_type)

                    samples_out.append({
                        "sample_idx":      r,
                        "completion":      completion,
                        "predicted":       predicted,
                        "correct":         correct,
                        "finish_reason":   finish_reason,
                        "n_tokens":        n_tokens,
                        "sum_logprob":     sum_logprob,
                        "mean_logprob":    mean_logprob,
                        "smc_log_weight":  smc_log_weight,
                        "smc_norm_weight": smc_norm_weight,
                        "smc_n_resamples": out.n_resamples,
                    })

                del outputs
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
                    "power_smc": {
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

        print(f"\n[POWER-SMC]  Overall   = {acc:.4f}  ({total_correct}/{total_samples})")
        print(f"[POWER-SMC]  Closed    = {closed_acc:.4f}  ({closed_correct}/{closed_total})")
        print(f"[POWER-SMC]  Open      = {open_acc:.4f}  ({open_correct}/{open_total})")

        fout.write(json.dumps({
            "_type":       "summary",
            "model":       args.model,
            "dataset":     args.dataset,
            "n_questions":  len(dataset),
            "start_index": args.start_index,
            "n_rollouts":  args.n_rollouts,
            "power_smc_config": {
                "alpha":             args.alpha,
                "n_particles":       args.n_particles,
                "kappa":             args.kappa,
                "resample":          args.resample,
                "prompt_batch_size": args.prompt_batch_size,
            },
            "summary": {
                "power_smc": {
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
    p = argparse.ArgumentParser(description="Medical VQA Power-SMC eval")

    p.add_argument("--dataset",                 type=str,   default="data/vqa_rad/VQA_RAD_test.json")
    p.add_argument("--model",                   type=str,   default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--model-type",             type=str,   default=None,
                   choices=["qwen", "llava", "medgemma"],
                   help="VLM type (auto-detected from model name if omitted).")
    p.add_argument("--output",                  type=str,   default="results/medical_power_smc.jsonl")
    p.add_argument("--n_rollouts",              type=int,   default=1,
                   help="Total independent SMC completions per question.")
    p.add_argument("--prompt_batch_size",       type=int,   default=1,
                   help="SMC runs processed in parallel per question.")
    p.add_argument("--max_new_tokens",          type=int,   default=256)
    p.add_argument("--start_index",             type=int,   default=0)
    p.add_argument("--max_examples",            type=int,   default=None)
    p.add_argument("--dtype",                   type=str,   default="bfloat16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--cot",    action="store_true",  default=True)
    p.add_argument("--no-cot", dest="cot",           action="store_false")
    p.add_argument("--resume", action="store_true")

    # Power-SMC hyperparameters
    p.add_argument("--alpha",       type=float, default=4.0)
    p.add_argument("--n_particles", type=int,   default=64)
    p.add_argument("--kappa",       type=float, default=0.5)
    p.add_argument("--resample",    type=str,   default="systematic",
                   choices=["systematic", "multinomial"])

    args = p.parse_args()
    run_eval(args)
