"""
MATH500 Eval — Power-SMC sampler
Same structure as eval_temp.py but uses PowerSMC from power_smc.py.
No force-answer, no chat template, same JSON dataset format.
Output: JSONL with methods: {power_smc}

Usage:
    python eval_power_smc.py \
        --dataset          data/MATH500.json \
        --model            Qwen/Qwen2.5-7B \
        --output           results/power_smc.jsonl \
        --alpha            4.0 \
        --n_particles      64 \
        --n_rollouts       32 \
        --prompt_batch_size 4

Resume a crashed run (skips already-written problems, appends new ones):
    python eval_power_smc.py ... --resume
"""
import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from grader import grade_answer, parse_answer, parse_boxed_space_form
from power_smc import PowerSMC, normalize_weights

# ─── prompt (identical to eval_temp.py) ────────────────────────────────────────
PROMPT      = "Can you solve the following math problem? "
COT         = " Please reason step by step, and put your final answer within \\boxed{}."
BASE_SUFFIX = " Put your final answer within \\boxed{}."


def build_prompt(question: str, cot: bool = True) -> str:
    return PROMPT + question + (COT if cot else BASE_SUFFIX)


def extract_answer(text: str):
    ans = parse_answer(text)
    if ans is None:
        ans = parse_boxed_space_form(text)
    return ans


def pass_at_k(n: int, c: int, k: int):
    if n < k:     return None
    if c == 0:    return 0.0
    if n - c < k: return 1.0
    prob = 1.0
    for i in range(k):
        prob *= (n - c - i) / (n - i)
    return 1.0 - prob


# ─── Resume helpers ─────────────────────────────────────────────────────────────
def load_completed(output_path: str) -> tuple[set, int, int]:
    """
    Read an existing JSONL output file and return:
      completed_indices : set of absolute problem idx values already written
      total_correct     : sum of n_correct across completed problems
      total_samples     : sum of n_rollouts across completed problems
    Silently skips malformed lines and the summary record.
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
            psc = rec.get("methods", {}).get("power_smc", {})
            total_correct += psc.get("n_correct",  0)
            total_samples += psc.get("n_rollouts", 0)
    return completed, total_correct, total_samples


# ─── Main ───────────────────────────────────────────────────────────────────────
def run_eval(args):
    # ── dataset ────────────────────────────────────────────────────────────────
    with open(args.dataset) as f:
        dataset = json.load(f)
    start = args.start_index
    end   = start + args.max_examples if args.max_examples is not None else len(dataset)
    dataset = dataset[start:end]
    print(f"Loaded {len(dataset)} problems from {args.dataset} (idx {start}–{end-1})")

    prompts = [build_prompt(d["prompt"], cot=args.cot) for d in dataset]

    # ── resume: discover already-completed problems ────────────────────────────
    completed_indices = set()
    total_correct     = 0
    total_samples     = 0
    out_mode          = "w"

    if args.resume:
        completed_indices, total_correct, total_samples = load_completed(args.output)
        if completed_indices:
            out_mode = "a"
            print(
                f"[resume] {len(completed_indices)} problems already in output — "
                f"skipping them ({total_correct}/{total_samples} correct so far)."
            )
        else:
            print("[resume] No existing output found — starting fresh.")

    # ── model ──────────────────────────────────────────────────────────────────
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    device    = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Pick the fastest available attention implementation:
    #   flash_attention_2 > sdpa > eager (default)
    # flash_attention_2 requires `pip install flash-attn` and Ampere+ GPU.
    # sdpa is built into PyTorch ≥2.0 and needs no extra install.
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"
    print(f"Using attention implementation: {attn_impl}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype_map[args.dtype],
        attn_implementation=attn_impl,
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    print(f"Model loaded on {device}")

    # ── Power-SMC sampler ──────────────────────────────────────────────────────
    sampler = PowerSMC(
        model           = model,
        tokenizer       = tokenizer,
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

    # ── output ─────────────────────────────────────────────────────────────────
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, out_mode, encoding="utf-8") as fout:
        for idx, data in enumerate(tqdm(dataset, desc="Problems")):
            abs_idx = start + idx          # absolute index in the full dataset

            # ── skip if already done (resume mode) ────────────────────────────
            if abs_idx in completed_indices:
                continue

            gt         = data["answer"]
            source     = data.get("source", "")
            prompt_str = prompts[idx]

            enc        = tokenizer(prompt_str, return_tensors="pt").to(device)
            input_ids  = enc["input_ids"]
            attn_mask  = enc.get("attention_mask")
            prompt_len = input_ids.shape[1]

            samples_out = []
            eos_id = tokenizer.eos_token_id

            for chunk_start in tqdm(range(0, args.n_rollouts, M), leave=False,
                                    desc=f"  idx={abs_idx} chunks"):
                actual_M = min(M, args.n_rollouts - chunk_start)

                # ── batched SMC: actual_M independent runs for this prompt ────
                if actual_M > 1:
                    outputs = sampler.generate_batch(
                        [input_ids] * actual_M,
                        [attn_mask] * actual_M,
                        max_new_tokens=args.max_new_tokens,
                    )
                else:
                    outputs = [sampler.generate(
                        input_ids, attn_mask, max_new_tokens=args.max_new_tokens
                    )]

                for k, out in enumerate(outputs):
                    r = chunk_start + k

                    # ── decode chosen particle ────────────────────────────────
                    chosen_i   = out.chosen_idx
                    gen_tokens = out.sequences[0, prompt_len:]

                    # trim at first EOS
                    eos_pos = (gen_tokens == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_pos) > 0:
                        gen_tokens    = gen_tokens[:int(eos_pos[0].item()) + 1]
                        finish_reason = "stop"
                    else:
                        finish_reason = "length"

                    n_tokens        = int(gen_tokens.shape[0])
                    completion      = tokenizer.decode(gen_tokens.cpu().tolist(), skip_special_tokens=True)
                    sum_logprob     = out.chosen_sum_logprob
                    mean_logprob    = sum_logprob / max(n_tokens, 1)
                    smc_log_weight  = float(out.log_weights[chosen_i].item())
                    smc_norm_weight = float(normalize_weights(out.log_weights)[chosen_i].item())

                    predicted = extract_answer(completion)
                    correct   = grade_answer(predicted, gt)

                    samples_out.append({
                        "sample_idx":      r,
                        "completion":      completion,
                        "predicted":       predicted,
                        "correct":         correct,
                        "finish_reason":   finish_reason,
                        "forced_answer":   False,
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

            # ── write + flush immediately so the record is on disk ────────────
            fout.write(json.dumps({
                "idx":          abs_idx,
                "question":     data["prompt"],
                "ground_truth": gt,
                "source":       source,
                "methods": {
                    "power_smc": {
                        "samples":    samples_out,
                        "n_correct":  c,
                        "n_rollouts": n,
                        "pass_at_k":  {str(k): pass_at_k(n, c, k) for k in range(1, n + 1)},
                    }
                },
            }) + "\n")
            fout.flush()            # Python buffer → OS
            os.fsync(fout.fileno()) # OS buffer → disk

        # ── summary ───────────────────────────────────────────────────────────
        acc = total_correct / max(total_samples, 1)
        print(f"\n[POWER-SMC]  pass@1 = {acc:.4f}  ({total_correct}/{total_samples})")
        fout.write(json.dumps({
            "_type":       "summary",
            "model":       args.model,
            "dataset":     args.dataset,
            "n_problems":  len(dataset),
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
                    "pass_at_1": acc,
                    "n_correct": total_correct,
                    "n_total":   total_samples,
                }
            },
        }) + "\n")
        fout.flush()

    print(f"Results written to {args.output}")


# ─── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MATH500 Power-SMC eval")

    # ── same as eval_temp.py ──────────────────────────────────────────────────
    p.add_argument("--dataset",                 type=str,   default="data/MATH500.json")
    p.add_argument("--model",                   type=str,   default="Qwen/Qwen2.5-7B")
    p.add_argument("--output",                  type=str,   default="results/power_smc.jsonl")
    p.add_argument("--n_rollouts",              type=int,   default=1,
                   help="Total independent SMC completions per problem.")
    p.add_argument("--prompt_batch_size",       type=int,   default=1,
                   help="SMC runs processed in parallel per problem (uses generate_batch). "
                        "n_rollouts must be divisible by this for even chunks.")
    p.add_argument("--max_new_tokens",          type=int,   default=2048)
    p.add_argument("--start_index",             type=int,   default=0,
                   help="First problem index (inclusive) — for splitting dataset across jobs.")
    p.add_argument("--max_examples",            type=int,   default=None,
                   help="Number of problems to run from start_index. Omit for all remaining.")
    p.add_argument("--dtype",                   type=str,   default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--cot",    action="store_true",  default=True)
    p.add_argument("--no-cot", dest="cot",           action="store_false")
    p.add_argument("--resume", action="store_true",
                   help="Read existing output file, skip already-completed problems, "
                        "and append new results. Safe to run multiple times.")

    # ── Power-SMC hyperparameters ─────────────────────────────────────────────
    p.add_argument("--alpha",       type=float, default=4.0,
                   help="Power exponent α. Paper MATH500: 4.0")
    p.add_argument("--n_particles", type=int,   default=64,
                   help="SMC particles N. Paper: 64")
    p.add_argument("--kappa",       type=float, default=0.5,
                   help="ESS resample threshold κ ∈ (0,1). Paper: 0.5")
    p.add_argument("--resample",    type=str,   default="systematic",
                   choices=["systematic", "multinomial"])

    args = p.parse_args()
    run_eval(args)
