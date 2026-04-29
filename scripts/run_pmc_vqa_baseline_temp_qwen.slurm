#!/bin/bash
#SBATCH --job-name=pmc_vqa_baseline_temp_qwen
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --exclude=evc40,evc39
#SBATCH --mail-user=erick.castillo@ucf.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Temperature baseline (CoT, T=1.0, 64 rollouts) — Qwen2.5-VL.
# 64 rollouts gives meaningful pass@k stats and a fair coverage comparison
# vs Power-SMC's 4 rollouts × 32 particles = ~128 implicit explored paths.
# Sequential rollouts via HF generate(); ~9-12 h on H100 with FA2 at this scale.

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

python -m eval.medical.eval_medical_baseline \
    --dataset data/pmc_vqa/PMC_VQA_test_clean.json \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results/pmc_vqa_qwen_baseline_temp_cot_n64.jsonl \
    --dtype bfloat16 \
    --temperature 1.0 \
    --n_rollouts 64 \
    --max_new_tokens 512 \
    --max_examples 100

END_TS=$(date +%s)
echo ""
echo "=== Job ended: $(date) ==="
ELAPSED=$((END_TS - START_TS))
printf "=== Total runtime: %dh %dm %ds ===\n" \
    $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60)) $((ELAPSED % 60))
