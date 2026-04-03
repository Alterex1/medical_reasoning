#!/bin/bash
#SBATCH --job-name=power_smc_math500
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
conda activate medProj

# ── Build flash-attn if not already installed ─────────────────────────────────
python -c "import flash_attn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "flash-attn not found, building..."
    TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=16 pip install flash-attn --no-build-isolation
    echo "flash-attn install done"
else
    echo "flash-attn already installed"
fi

# ── Run eval ──────────────────────────────────────────────────────────────────
echo "Starting Power-SMC eval..."
date

python -m eval.math.eval_power_smc \
    --dataset data/MATH500.json \
    --model Qwen/Qwen2.5-7B \
    --output results/power_smc.jsonl \
    --dtype bfloat16 \
    --alpha 4.0 \
    --n_particles 64 \
    --n_rollouts 32 \
    --prompt_batch_size 4 \
    --max_new_tokens 512

echo "Done"
date
