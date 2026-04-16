#!/bin/bash
#SBATCH --job-name=medgemma_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1

# Med-Gemma (google/medgemma-4b-it) evaluation on VQA-RAD.
# Runs all 3 methods: Power-SMC, greedy baseline, temperature baseline.
# Also runs Power-SMC + greedy with --no-cot for direct-prompt comparison.

hostname; pwd; date

# ── Environment ───────────────────────────────────────────────────────────────
module restore
module list

conda activate medProj

MODEL="google/medgemma-4b-it"
DATASET="data/vqa_rad/VQA_RAD_test.json"
DTYPE="bfloat16"
MAX_TOKENS=512

# ── 1. Power-SMC with CoT ───────────────────────────────────────────────────
echo "=== Med-Gemma Power-SMC (CoT) ==="
date
python -m eval.medical.eval_medical_smc \
    --dataset $DATASET \
    --model $MODEL \
    --output results/medgemma_power_smc_cot.jsonl \
    --dtype $DTYPE \
    --alpha 4.0 \
    --n_particles 16 \
    --n_rollouts 4 \
    --prompt_batch_size 1 \
    --max_new_tokens $MAX_TOKENS

# ── 2. Greedy baseline with CoT ─────────────────────────────────────────────
echo "=== Med-Gemma Greedy baseline (CoT) ==="
date
python -m eval.medical.eval_medical_baseline \
    --dataset $DATASET \
    --model $MODEL \
    --output results/medgemma_baseline_greedy_cot.jsonl \
    --dtype $DTYPE \
    --temperature 0.0 \
    --max_new_tokens $MAX_TOKENS

# ── 3. Temperature baseline with CoT ────────────────────────────────────────
echo "=== Med-Gemma Temperature baseline (CoT) ==="
date
python -m eval.medical.eval_medical_baseline \
    --dataset $DATASET \
    --model $MODEL \
    --output results/medgemma_baseline_temp_cot.jsonl \
    --dtype $DTYPE \
    --temperature 1.0 \
    --n_rollouts 4 \
    --max_new_tokens $MAX_TOKENS

# ── 4. Greedy baseline, direct prompt (no CoT) ──────────────────────────────
echo "=== Med-Gemma Greedy baseline (direct / no-CoT) ==="
date
python -m eval.medical.eval_medical_baseline \
    --dataset $DATASET \
    --model $MODEL \
    --output results/medgemma_baseline_greedy_direct.jsonl \
    --dtype $DTYPE \
    --temperature 0.0 \
    --max_new_tokens $MAX_TOKENS \
    --no-cot

# ── 5. Power-SMC, direct prompt (no CoT) ────────────────────────────────────
echo "=== Med-Gemma Power-SMC (direct / no-CoT) ==="
date
python -m eval.medical.eval_medical_smc \
    --dataset $DATASET \
    --model $MODEL \
    --output results/medgemma_power_smc_direct.jsonl \
    --dtype $DTYPE \
    --alpha 4.0 \
    --n_particles 16 \
    --n_rollouts 4 \
    --prompt_batch_size 1 \
    --max_new_tokens $MAX_TOKENS \
    --no-cot

echo "=== All Med-Gemma evals done ==="
date
