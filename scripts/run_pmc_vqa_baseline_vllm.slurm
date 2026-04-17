#!/bin/bash
#SBATCH --job-name=pmc_vqa_baseline_vllm
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

# Use the dedicated vLLM env (separate from medProj because vLLM pins its
# own torch/transformers versions — see eval_medical_baseline_vllm.py docstring).
conda activate medProj-vllm

# ── Run: PMC-VQA baseline with vLLM, T=1.0, 32 rollouts/question ─────────────
echo "=== vLLM baseline eval (Qwen2.5-VL, T=1.0, n=32) ==="
date

python -m eval.medical.eval_medical_baseline_vllm \
    --dataset data/pmc_vqa/PMC_VQA_test_clean.json \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results/pmc_vqa_baseline_vllm_temp1_n32.jsonl \
    --dtype bfloat16 \
    --temperature 1.0 \
    --n_rollouts 32 \
    --max_new_tokens 512 \
    --max_model_len 4096 \
    --gpu_memory_utilization 0.9 \
    --tensor_parallel_size 1

echo "=== Done ==="
date
