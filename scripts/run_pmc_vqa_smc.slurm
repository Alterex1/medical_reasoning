#!/bin/bash
#SBATCH --job-name=pmc_vqa_smc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --mail-user=erick.castillo@ucf.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Power-SMC (CoT) on PMC-VQA test_clean, first 100 questions, three models:
#   1. Qwen2.5-VL-7B-Instruct
#   2. LLaVA-Med-v1.5-mistral-7b
#   3. Med-Gemma 4b-it
# No baselines, no direct-prompt variants.

START_TS=$(date +%s)
echo "=== Job started: $(date) ==="
hostname; pwd

# ── Environment ───────────────────────────────────────────────────────────────
module restore
module list

conda activate medProj

DATASET="data/pmc_vqa/PMC_VQA_test_clean.json"
DTYPE="bfloat16"
MAX_TOKENS=512
MAX_EXAMPLES=100

# Power-SMC config matches prior VQA-RAD runs (alpha=4, N=16, kappa=0.5,
# n_rollouts=4) so PMC-VQA results are directly comparable.
ALPHA=4.0
N_PARTICLES=16
N_ROLLOUTS=4
PROMPT_BATCH=1

# ── 1. Qwen2.5-VL ───────────────────────────────────────────────────────────
echo ""
echo "=== [1/3] Qwen2.5-VL Power-SMC (CoT) ==="
date
python -m eval.medical.eval_medical_smc \
    --dataset $DATASET \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results/pmc_vqa_qwen_power_smc_cot.jsonl \
    --dtype $DTYPE \
    --alpha $ALPHA \
    --n_particles $N_PARTICLES \
    --n_rollouts $N_ROLLOUTS \
    --prompt_batch_size $PROMPT_BATCH \
    --max_new_tokens $MAX_TOKENS \
    --max_examples $MAX_EXAMPLES

# ── 2. LLaVA-Med ────────────────────────────────────────────────────────────
echo ""
echo "=== [2/3] LLaVA-Med Power-SMC (CoT) ==="
date
python -m eval.medical.eval_medical_smc \
    --dataset $DATASET \
    --model microsoft/llava-med-v1.5-mistral-7b \
    --output results/pmc_vqa_llava_med_power_smc_cot.jsonl \
    --dtype $DTYPE \
    --alpha $ALPHA \
    --n_particles $N_PARTICLES \
    --n_rollouts $N_ROLLOUTS \
    --prompt_batch_size $PROMPT_BATCH \
    --max_new_tokens $MAX_TOKENS \
    --max_examples $MAX_EXAMPLES

# ── 3. Med-Gemma ────────────────────────────────────────────────────────────
echo ""
echo "=== [3/3] Med-Gemma Power-SMC (CoT) ==="
date
python -m eval.medical.eval_medical_smc \
    --dataset $DATASET \
    --model google/medgemma-4b-it \
    --output results/pmc_vqa_medgemma_power_smc_cot.jsonl \
    --dtype $DTYPE \
    --alpha $ALPHA \
    --n_particles $N_PARTICLES \
    --n_rollouts $N_ROLLOUTS \
    --prompt_batch_size $PROMPT_BATCH \
    --max_new_tokens $MAX_TOKENS \
    --max_examples $MAX_EXAMPLES

END_TS=$(date +%s)
echo ""
echo "=== Job ended: $(date) ==="
ELAPSED=$((END_TS - START_TS))
printf "=== Total runtime: %dh %dm %ds ===\n" \
    $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60)) $((ELAPSED % 60))
