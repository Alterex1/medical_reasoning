#!/bin/bash
#SBATCH --job-name=pmc_smc_v100_llava
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:tesla_v100-pcie-32gb:1
#SBATCH --mail-user=erick.castillo@ucf.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Power-SMC (CoT) on PMC-VQA test_clean — first 100 questions, LLaVA-Med.
# V100 fallback (32GB, no flash-attn). Standalone per-model job.
#
# BF16 for cross-model comparability with the Qwen / Med-Gemma runs.
# V100 has no BF16 tensor cores so this runs via emulation — slower than
# FP16 but matches the dtype used by the other V100 runs.

START_TS=$(date +%s)
echo "=== Job started: $(date) ==="
hostname; pwd

module restore
module list

conda activate medProj

python -m eval.medical.eval_medical_smc \
    --dataset data/pmc_vqa/PMC_VQA_test_clean.json \
    --model microsoft/llava-med-v1.5-mistral-7b \
    --output results/pmc_vqa_llava_med_power_smc_cot_v100.jsonl \
    --dtype bfloat16 \
    --alpha 4.0 \
    --n_particles 16 \
    --n_rollouts 4 \
    --prompt_batch_size 1 \
    --max_new_tokens 512 \
    --max_examples 100

END_TS=$(date +%s)
echo ""
echo "=== Job ended: $(date) ==="
ELAPSED=$((END_TS - START_TS))
printf "=== Total runtime: %dh %dm %ds ===\n" \
    $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60)) $((ELAPSED % 60))
