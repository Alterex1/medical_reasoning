"""
Medical VQA Eval — vLLM backend (greedy / temperature sampling)

Drop-in replacement for ``eval_medical_baseline.py`` that uses vLLM instead
of HuggingFace transformers. Same prompts, same grading, same JSONL output
format, so results from the two backends are directly comparable.

Why vLLM:
  - Continuous batching + paged KV cache → much higher throughput on long runs.
  - Native multi-sample requests via ``SamplingParams(n=K)`` — one forward pass
    over the prompt's KV cache produces K rollouts (no per-rollout reprefill).
  - Built-in ``finish_reason`` ("stop" / "length") on every output.

Currently supports Qwen2.5-VL only (vLLM has first-class support). LLaVA-Med
and Med-Gemma run through the HF baseline only.

Usage (from repo root):
    # Greedy on PMC-VQA test_clean
    python -m eval.medical.eval_medical_baseline_vllm \
        --dataset       data/pmc_vqa/PMC_VQA_test_clean.json \
        --model         Qwen/Qwen2.5-VL-7B-Instruct \
        --output        results/medical_baseline_vllm_greedy.jsonl \
        --dtype         bfloat16 \
        --temperature   0.0 \
        --max_new_tokens 512

    # Temperature sampling, 4 rollouts/question (single vLLM request per Q)
    python -m eval.medical.eval_medical_baseline_vllm \
        --dataset       data/pmc_vqa/PMC_VQA_test_clean.json \
        --model         Qwen/Qwen2.5-VL-7B-Instruct \
        --output        results/medical_baseline_vllm_temp.jsonl \
        --temperature   1.0 \
        --n_rollouts    4

    # Resume a crashed run
    python -m eval.medical.eval_medical_baseline_vllm ... --resume
"""
import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

from eval.medical.medical_grader import (
    grade_medical_answer,
    is_closed_ended,
    parse_medical_answer,
    parse_mcq_answer,
)
from eval.medical.model_adapters import MEDICAL_SYSTEM_PROMPT, get_prompt
from eval.medical.eval_medical_smc import load_completed, pass_at_k


# ─── Message construction ─────────────────────────────────────────────────────

def _build_messages(image_path: str, prompt_text: str) -> list:
    """Build OpenAI-format chat messages with an image_url for vLLM.

    vLLM's ``LLM.chat()`` reads ``image_url`` from each user-content block,
    fetches the image (file://, http(s)://, or data:), runs it through the
    model's processor, and applies the chat template internally. So the
    eval doesn't need to touch the processor or chat template directly.
    """
    abs_path = os.path.abspath(image_path).replace("\\", "/")
    return [
        {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"file://{abs_path}"}},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_eval(args):
    if not Path(args.dataset).exists():
        print(f"Dataset not found: {args.dataset}")
        print("Run: python data/download_pmc_vqa.py")
        sys.exit(1)

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

    # ── vLLM engine ───────────────────────────────────────────────────────────
    # Imported lazily so the module is importable on systems without vLLM
    # (e.g. for unit testing the message-construction helpers).
    from vllm import LLM, SamplingParams

    print(f"Loading {args.model} (vLLM) ...")
    llm_kwargs = dict(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed if args.seed is not None else 0,
    )
    # Match the HF QwenVLAdapter's image-resize config so per-image token
    # counts (and therefore the visible content) are identical between backends.
    if "qwen" in args.model.lower():
        llm_kwargs["mm_processor_kwargs"] = {
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        }
    llm = LLM(**llm_kwargs)
    print("vLLM engine ready")

    # ── sampling params ───────────────────────────────────────────────────────
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p if args.temperature > 0 else 1.0,
        max_tokens=args.max_new_tokens,
        n=args.n_rollouts,
        seed=args.seed,
    )
    print(
        f"\nSampling: temperature={args.temperature}  n={args.n_rollouts}  "
        f"max_tokens={args.max_new_tokens}\n"
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

            prompt_text = get_prompt(question, cot=args.cot, question_type=q_type)
            messages    = _build_messages(image_path, prompt_text)

            # Single vLLM request — n=n_rollouts samples come back together,
            # sharing one prefill of the (image + prompt) KV cache.
            request_outputs = llm.chat(
                messages,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            ro = request_outputs[0]

            samples_out = []
            for r, comp in enumerate(ro.outputs):
                completion    = comp.text
                n_tokens      = len(comp.token_ids)
                finish_reason = comp.finish_reason or "unknown"

                if q_type == "mcq":
                    predicted = parse_mcq_answer(completion,
                                                 choices=data.get("choices"))
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
                "choices":      data.get("choices"),
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
        acc        = total_correct / max(total_samples, 1)
        closed_acc = closed_correct / max(closed_total, 1)
        open_acc   = open_correct   / max(open_total, 1)

        print(f"\n[VLLM]  Overall = {acc:.4f}  ({total_correct}/{total_samples})")
        print(f"[VLLM]  Closed  = {closed_acc:.4f}  ({closed_correct}/{closed_total})")
        print(f"[VLLM]  Open    = {open_acc:.4f}  ({open_correct}/{open_total})")

        fout.write(json.dumps({
            "_type":       "summary",
            "model":       args.model,
            "dataset":     args.dataset,
            "n_questions": len(dataset),
            "start_index": args.start_index,
            "n_rollouts":  args.n_rollouts,
            "backend":     "vllm",
            "baseline_config": {
                "temperature": args.temperature,
                "top_p":       args.top_p,
                "max_tokens":  args.max_new_tokens,
                "seed":        args.seed,
            },
            "summary": {
                "baseline": {
                    "overall_accuracy": acc,
                    "closed_accuracy":  closed_acc,
                    "open_accuracy":    open_acc,
                    "n_correct":        total_correct,
                    "n_total":          total_samples,
                    "closed_correct":   closed_correct,
                    "closed_total":     closed_total,
                    "open_correct":     open_correct,
                    "open_total":       open_total,
                }
            },
        }) + "\n")
        fout.flush()

    print(f"Results written to {args.output}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Medical VQA baseline eval (vLLM backend)")

    p.add_argument("--dataset",        type=str,
                   default="data/pmc_vqa/PMC_VQA_test_clean.json")
    p.add_argument("--model",          type=str,
                   default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--output",         type=str,
                   default="results/medical_baseline_vllm.jsonl")
    p.add_argument("--n_rollouts",     type=int,   default=1,
                   help="Samples per question. With temperature=0.0, vLLM "
                        "still returns n copies of the greedy decode.")
    p.add_argument("--max_new_tokens", type=int,   default=512)
    p.add_argument("--max_model_len",  type=int,   default=4096,
                   help="Max prompt + generation length (vLLM KV budget).")
    p.add_argument("--start_index",    type=int,   default=0)
    p.add_argument("--max_examples",   type=int,   default=None)
    p.add_argument("--dtype",          type=str,   default="bfloat16",
                   choices=["float16", "bfloat16", "float32", "auto"])
    p.add_argument("--temperature",    type=float, default=0.0,
                   help="0.0 = greedy.")
    p.add_argument("--top_p",          type=float, default=1.0)
    p.add_argument("--seed",           type=int,   default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--tensor_parallel_size",   type=int,   default=1)
    p.add_argument("--cot",    action="store_true",  default=True)
    p.add_argument("--no-cot", dest="cot",           action="store_false")
    p.add_argument("--resume", action="store_true")

    args = p.parse_args()
    run_eval(args)
