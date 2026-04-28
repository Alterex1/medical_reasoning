#!/bin/bash
#SBATCH --job-name=smoke_dtype_medgemma
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
#SBATCH --mail-user=erick.castillo@ucf.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Smoke test: run Med-Gemma Power-SMC on a single PMC-VQA question with
# each candidate dtype. Logs success/failure and wall-time per dtype so
# we can pick the fastest dtype that actually works on V100.
#
# Why Med-Gemma only:
#   - Qwen 7B: FP16 already works. FP32 won't fit in 32GB. BF16 on V100
#     is emulated (slow) — no upside over FP16.
#   - LLaVA-Med 7B: same as Qwen.
#   - Med-Gemma 4B: FP16 crashed on the real run (multinomial assert from
#     softmax underflow). FP32 should work but is slow. BF16 is
#     emulated on V100 — does its wider range still fix the underflow?
#     This script tells us empirically.

START_TS=$(date +%s)
echo "=== Job started: $(date) ==="
hostname; pwd
nvidia-smi --query-gpu=name,memory.total --format=csv

module restore
module list

conda activate medProj

DATASET="data/pmc_vqa/PMC_VQA_test_clean.json"
MODEL="google/medgemma-4b-it"

run_smoke() {
    local dtype=$1
    local out="results/smoke_medgemma_${dtype}.jsonl"
    local t_start=$(date +%s)

    echo ""
    echo "===================================================================="
    echo "[$dtype] Med-Gemma smoke (1 question, 1 rollout)"
    echo "===================================================================="
    date

    # --max_examples 1 --n_rollouts 1 — one rollout on the first question
    python -m eval.medical.eval_medical_smc \
        --dataset $DATASET \
        --model $MODEL \
        --output $out \
        --dtype $dtype \
        --alpha 4.0 \
        --n_particles 16 \
        --n_rollouts 1 \
        --prompt_batch_size 1 \
        --max_new_tokens 512 \
        --max_examples 1
    local rc=$?

    local t_end=$(date +%s)
    local elapsed=$((t_end - t_start))
    if [ $rc -eq 0 ]; then
        echo "[$dtype] OK in ${elapsed}s"
    else
        echo "[$dtype] FAILED (exit $rc) after ${elapsed}s"
    fi
}

# Run all three dtypes — capture exit codes, don't bail on a failure
set +e
run_smoke float16
run_smoke float32
run_smoke bfloat16

echo ""
echo "===================================================================="
echo "SUMMARY"
echo "===================================================================="
for dtype in float16 float32 bfloat16; do
    out="results/smoke_medgemma_${dtype}.jsonl"
    if [ -s "$out" ]; then
        # Pull the predicted answer + token count out of the JSONL
        python - <<PY
import json, sys
try:
    rec = json.loads(open("$out").readline())
    s = rec["methods"]["power_smc"]["samples"][0]
    print(f"[$dtype] OK | predicted={s['predicted']!r:>10} | "
          f"correct={s['correct']} | tokens={s['n_tokens']:>3} | "
          f"gt={rec['ground_truth']!r}")
except Exception as e:
    print(f"[$dtype] output unreadable: {e}")
PY
    else
        echo "[$dtype] no output (job crashed or file empty)"
    fi
done

END_TS=$(date +%s)
echo ""
echo "=== Job ended: $(date) ==="
ELAPSED=$((END_TS - START_TS))
printf "=== Total runtime: %dh %dm %ds ===\n" \
    $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60)) $((ELAPSED % 60))
