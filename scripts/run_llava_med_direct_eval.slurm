#!/bin/bash
#SBATCH --job-name=llava_med_direct
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1

# Direct-prompt (no-CoT) comparison for LLaVA-Med.
# Previous run used CoT prompts and got baseline=21.9%, SMC=33.3%.
# LLaVA-Med-v1.5 was instruction-tuned for direct Q→A, so CoT may be
# hurting both methods. This job runs greedy + Power-SMC with --no-cot
# to verify Power-SMC's advantage holds under LLaVA-Med's native prompt
# style.

hostname; pwd; date

# ── Environment ───────────────────────────────────────────────────────────────
module restore
module list

conda activate medProj

MODEL="microsoft/llava-med-v1.5-mistral-7b"
DATASET="data/vqa_rad/VQA_RAD_test.json"
DTYPE="bfloat16"
MAX_TOKENS=512

# ── 1. Greedy baseline, direct prompt ────────────────────────────────────────
echo "=== LLaVA-Med Greedy baseline (direct / no-CoT) ==="
date
python -m eval.medical.eval_medical_baseline \
    --dataset $DATASET \
    --model $MODEL \
    --output results/llava_med_baseline_greedy_direct.jsonl \
    --dtype $DTYPE \
    --temperature 0.0 \
    --max_new_tokens $MAX_TOKENS \
    --no-cot

# ── 2. Power-SMC, direct prompt ──────────────────────────────────────────────
echo "=== LLaVA-Med Power-SMC (direct / no-CoT) ==="
date
python -m eval.medical.eval_medical_smc \
    --dataset $DATASET \
    --model $MODEL \
    --output results/llava_med_power_smc_direct.jsonl \
    --dtype $DTYPE \
    --alpha 4.0 \
    --n_particles 16 \
    --n_rollouts 4 \
    --prompt_batch_size 1 \
    --max_new_tokens $MAX_TOKENS \
    --no-cot

echo "=== All LLaVA-Med direct-prompt evals done ==="
date
