#!/bin/bash
#SBATCH --job-name=llava_med_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1

hostname; pwd; date

# ── Environment ───────────────────────────────────────────────────────────────
module restore
module list

conda activate medProj

MODEL="microsoft/llava-med-v1.5-mistral-7b"
DATASET="data/vqa_rad/VQA_RAD_test.json"
DTYPE="bfloat16"
MAX_TOKENS=512

# ── 1. Power-SMC (full test set) ─────────────────────────────────────────────
echo "=== LLaVA-Med Power-SMC eval ==="
date
python -m eval.medical.eval_medical_smc \
    --dataset $DATASET \
    --model $MODEL \
    --output results/llava_med_power_smc_full.jsonl \
    --dtype $DTYPE \
    --alpha 4.0 \
    --n_particles 16 \
    --n_rollouts 4 \
    --prompt_batch_size 1 \
    --max_new_tokens $MAX_TOKENS

# ── 2. Greedy baseline (full test set) ───────────────────────────────────────
echo "=== LLaVA-Med Greedy baseline eval ==="
date
python -m eval.medical.eval_medical_baseline \
    --dataset $DATASET \
    --model $MODEL \
    --output results/llava_med_baseline_greedy_full.jsonl \
    --dtype $DTYPE \
    --temperature 0.0 \
    --max_new_tokens $MAX_TOKENS

# ── 3. Temperature baseline (full test set) ──────────────────────────────────
echo "=== LLaVA-Med Temperature baseline eval ==="
date
python -m eval.medical.eval_medical_baseline \
    --dataset $DATASET \
    --model $MODEL \
    --output results/llava_med_baseline_temp_full.jsonl \
    --dtype $DTYPE \
    --temperature 1.0 \
    --n_rollouts 4 \
    --max_new_tokens $MAX_TOKENS

echo "=== All LLaVA-Med evals done ==="
date
