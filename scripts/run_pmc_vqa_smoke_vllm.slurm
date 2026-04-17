#!/bin/bash
#SBATCH --job-name=pmc_vqa_smoke_vllm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --mail-user=erick.castillo@ucf.edu
#SBATCH --mail-type=BEGIN,END,FAIL

START_TS=$(date +%s)
echo "=== Job started: $(date) ==="
hostname; pwd

# ── Environment ───────────────────────────────────────────────────────────────
module restore
module list

conda activate medProj-vllm

# ── Smoke test: 10 questions, T=1.0, 32 rollouts/question ────────────────────
# Purpose: verify the vLLM pipeline (model load, image fetch, chat template,
# answer parsing, JSONL write) end-to-end before launching the full run.
# Expected runtime: ~5–15 min on H100 (model load dominates).
echo "=== vLLM smoke test (10 questions, T=1.0, n=32) ==="

python -m eval.medical.eval_medical_baseline_vllm \
    --dataset data/pmc_vqa/PMC_VQA_test_clean.json \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results/pmc_vqa_smoke_vllm.jsonl \
    --dtype bfloat16 \
    --temperature 1.0 \
    --n_rollouts 32 \
    --max_new_tokens 512 \
    --max_model_len 4096 \
    --gpu_memory_utilization 0.9 \
    --tensor_parallel_size 1 \
    --max_examples 10

echo "=== Smoke test done ==="

echo ""
echo "Quick sanity check — accuracy + per-question correctness:"
python - <<'PY'
import json
from pathlib import Path
path = Path("results/pmc_vqa_smoke_vllm.jsonl")
total_c = total_n = 0
for line in path.read_text().splitlines():
    rec = json.loads(line)
    if rec.get("_type") == "summary":
        s = rec["summary"]["baseline"]
        print(f"\nSUMMARY: overall={s['overall_accuracy']:.3f} "
              f"({s['n_correct']}/{s['n_total']})")
        continue
    b = rec["methods"]["baseline"]
    print(f"  idx={rec['idx']:>4}  gt={rec['ground_truth']}  "
          f"correct={b['n_correct']}/{b['n_rollouts']}  "
          f"pass@1={b['pass_at_k']['1']:.2f}")
PY

END_TS=$(date +%s)
echo ""
echo "=== Job ended: $(date) ==="
ELAPSED=$((END_TS - START_TS))
printf "=== Total runtime: %dh %dm %ds ===\n" \
    $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60)) $((ELAPSED % 60))
