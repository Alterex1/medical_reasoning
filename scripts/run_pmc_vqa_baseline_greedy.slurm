#!/bin/bash
#SBATCH --job-name=pmc_vqa_baseline_greedy
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc40,evc39
#SBATCH --requeue
#SBATCH --mail-user=erick.castillo@ucf.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Greedy baseline (CoT, T=0.0, 1 rollout) on PMC-VQA test_clean, first 100
# questions, all three models sequentially. Greedy is fast enough that one
# job covers all three (~30-60 min/model on H100 with FA2).

# Exit immediately on any command failure (otherwise a python crash
# still lets the script's trailing `printf` line run, and SLURM logs
# the job as COMPLETED 0:0 — silent failure).
set -e

START_TS=$(date +%s)
echo "=== Job started: $(date) ==="
hostname; pwd

module restore
module list
conda activate medProj

DATASET="data/pmc_vqa/PMC_VQA_test_clean.json"
DTYPE="bfloat16"
MAX_TOKENS=512
MAX_EXAMPLES=100

# ── 1. Qwen2.5-VL ───────────────────────────────────────────────────────────
echo ""
echo "=== [1/3] Qwen2.5-VL Greedy baseline (CoT) ==="
date
python -m eval.medical.eval_medical_baseline \
    --dataset $DATASET \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results/pmc_vqa_qwen_baseline_greedy_cot.jsonl \
    --dtype $DTYPE \
    --temperature 0.0 \
    --max_new_tokens $MAX_TOKENS \
    --max_examples $MAX_EXAMPLES \
    --resume

# ── 2. LLaVA-Med ────────────────────────────────────────────────────────────
echo ""
echo "=== [2/3] LLaVA-Med Greedy baseline (CoT) ==="
date
python -m eval.medical.eval_medical_baseline \
    --dataset $DATASET \
    --model microsoft/llava-med-v1.5-mistral-7b \
    --output results/pmc_vqa_llava_med_baseline_greedy_cot.jsonl \
    --dtype $DTYPE \
    --temperature 0.0 \
    --max_new_tokens $MAX_TOKENS \
    --max_examples $MAX_EXAMPLES \
    --resume

# ── 3. Med-Gemma ────────────────────────────────────────────────────────────
echo ""
echo "=== [3/3] Med-Gemma Greedy baseline (CoT) ==="
date
python -m eval.medical.eval_medical_baseline \
    --dataset $DATASET \
    --model google/medgemma-4b-it \
    --output results/pmc_vqa_medgemma_baseline_greedy_cot.jsonl \
    --dtype $DTYPE \
    --temperature 0.0 \
    --max_new_tokens $MAX_TOKENS \
    --max_examples $MAX_EXAMPLES \
    --resume

END_TS=$(date +%s)
echo ""
echo "=== Job ended: $(date) ==="
ELAPSED=$((END_TS - START_TS))
printf "=== Total runtime: %dh %dm %ds ===\n" \
    $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60)) $((ELAPSED % 60))
