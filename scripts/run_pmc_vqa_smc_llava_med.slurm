#!/bin/bash
#SBATCH --job-name=pmc_vqa_smc_llava_med
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

# Power-SMC (CoT) on PMC-VQA test_clean, first 100 questions.
# Model: LLaVA-Med
# No baselines, no direct-prompt variants.

# Exit immediately on any command failure (otherwise a python crash
# still lets the script's trailing `printf` line run, and SLURM logs
# the job as COMPLETED 0:0 — silent failure that bit us on evc40).
set -e

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

# Power-SMC config: alpha=4, N=32, kappa=0.5, n_rollouts=4.
# Bumped N from 16 -> 32 to give SMC more particles to steer with on
# PMC-VQA's MCQ format (more parallel reasoning paths per rollout).
# Wall-clock cost roughly doubles per rollout vs N=16.
ALPHA=4.0
N_PARTICLES=32
N_ROLLOUTS=4
PROMPT_BATCH=1

# ── LLaVA-Med ───────────────────────────────────────────────────────────────
echo ""
echo "=== LLaVA-Med Power-SMC (CoT) ==="
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
    --max_examples $MAX_EXAMPLES \
    --resume

END_TS=$(date +%s)
echo ""
echo "=== Job ended: $(date) ==="
ELAPSED=$((END_TS - START_TS))
printf "=== Total runtime: %dh %dm %ds ===\n" \
    $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60)) $((ELAPSED % 60))
