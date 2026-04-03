#!/bin/bash
#SBATCH --job-name=medical_vqa_eval
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

# ── Run all three evals sequentially ─────────────────────────────────────────

# 1. Power-SMC (full test set)
echo "=== Power-SMC eval ==="
date
python -m eval.medical.eval_medical_smc \
    --dataset data/vqa_rad/VQA_RAD_test.json \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results/medical_power_smc_full.jsonl \
    --dtype bfloat16 \
    --alpha 4.0 \
    --n_particles 16 \
    --n_rollouts 4 \
    --prompt_batch_size 1 \
    --max_new_tokens 512

# 2. Greedy baseline (full test set)
echo "=== Greedy baseline eval ==="
date
python -m eval.medical.eval_medical_baseline \
    --dataset data/vqa_rad/VQA_RAD_test.json \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results/medical_baseline_greedy_full.jsonl \
    --dtype bfloat16 \
    --temperature 0.0 \
    --max_new_tokens 512

# 3. Temperature baseline (full test set)
echo "=== Temperature baseline eval ==="
date
python -m eval.medical.eval_medical_baseline \
    --dataset data/vqa_rad/VQA_RAD_test.json \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results/medical_baseline_temp_full.jsonl \
    --dtype bfloat16 \
    --temperature 1.0 \
    --n_rollouts 4 \
    --max_new_tokens 512

echo "=== All evals done ==="
date
