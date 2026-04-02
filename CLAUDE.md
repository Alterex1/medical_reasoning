# CLAUDE.md

## Project Overview

Adapting Power-SMC (Power Sequential Monte Carlo) for **medical reasoning**. Power-SMC is a low-latency, training-free reasoning framework for LLMs from the paper "Power-SMC: Low-Latency Sequence-Level Power Sampling for Training-Free LLM Reasoning" (arXiv:2602.10273).

The existing implementation targets math reasoning (MATH500 benchmark). The goal of this project is to adapt it for medical reasoning tasks — this requires changes to the evaluation pipeline, grading logic, dataset, and potentially the prompting/scoring strategy.

## Repository Structure

Source files are in the **repository root**:

| File | Purpose |
|------|---------|
| `power_smc.py` | Core SMC sampling algorithm (`PowerSMC` class) |
| `eval_power_smc.py` | MATH500 benchmark evaluation harness |
| `grader.py` | Answer grading via SymPy simplification and LaTeX parsing |
| `math_normalize.py` | Mathematical expression normalization |
| `MATH500.json` | 500 math problems dataset with ground truth |
| `run_inference.slurm` | SLURM HPC job script (legacy) |
| `run_math500_eval.slurm` | SLURM job: build flash-attn + run full MATH500 eval |
| `install_flash_attn.slurm` | SLURM job: CPU-only flash-attn build |
| `requirements.txt` | Pinned Python dependencies |

## Tech Stack

- **Language**: Python 3
- **Core deps**: PyTorch (2.10+cu126), Transformers (5.4), NumPy, SymPy, math-verify, tqdm
- **Other notable deps**: latex2sympy2_extended, pandas, pyarrow, safetensors, rich, triton
- **Models**: HuggingFace pretrained models (e.g., `Qwen/Qwen2.5-7B`)
- **GPU**: CUDA 12.6 (nvidia libs pinned in requirements)

## Install Dependencies

```bash
pip install -r requirements.txt

# Flash Attention 2 (optional, requires Ampere+ GPU i.e. A100/H100/H200)
# Build from source — needs CUDA toolkit and takes 10-20 min with 16 CPUs
# For CPU-only build node, set the target arch explicitly:
TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=8 pip install flash-attn --no-build-isolation

# Or submit as a SLURM job:
sbatch install_flash_attn.slurm
```

If flash-attn is not installed, the code automatically falls back to PyTorch SDPA (no extra install needed).

## Running

```bash
# Evaluate on MATH500 (H100, bfloat16, flash-attn)
python eval_power_smc.py \
    --dataset MATH500.json \
    --model Qwen/Qwen2.5-7B \
    --output results/power_smc.jsonl \
    --dtype bfloat16 \
    --alpha 4.0 \
    --n_particles 64 \
    --n_rollouts 32 \
    --prompt_batch_size 4 \
    --max_new_tokens 512

# Quick test run (5 problems)
python eval_power_smc.py \
    --dataset MATH500.json \
    --model Qwen/Qwen2.5-7B \
    --output results/power_smc.jsonl \
    --dtype bfloat16 \
    --alpha 4.0 \
    --n_particles 64 \
    --n_rollouts 32 \
    --prompt_batch_size 4 \
    --max_new_tokens 512 \
    --max_examples 5

# V100 fallback (float16, reduced params to fit 32GB VRAM)
python eval_power_smc.py \
    --dataset MATH500.json \
    --model Qwen/Qwen2.5-7B \
    --output results/power_smc.jsonl \
    --dtype float16 \
    --alpha 4.0 \
    --n_particles 16 \
    --n_rollouts 4 \
    --prompt_batch_size 1 \
    --max_new_tokens 512

# Resume interrupted runs
python eval_power_smc.py ... --resume
```

## Key Parameters

- `--alpha` (default=4.0) — Power exponent for importance distribution
- `--n_particles` (default=64) — Number of SMC particles/samples
- `--kappa` (default=0.5) — ESS resampling threshold
- `--resample` ("systematic"|"multinomial") — Resampling method
- `--max_new_tokens` (default=2048) — Max generation length
- `--n_rollouts` (default=1) — Independent SMC runs per problem

## Adaptation: Math → Medical Reasoning

Key areas that need adaptation from the current math-focused implementation:

- **Dataset**: Replace MATH500 with a medical reasoning dataset (e.g., MedQA, PubMedQA, USMLE-style questions)
- **Evaluation/Grading**: `grader.py` and `math_normalize.py` are math-specific (SymPy, LaTeX parsing). Medical reasoning needs domain-appropriate grading (e.g., multiple-choice matching, clinical correctness)
- **Prompting**: The current prompts are math-oriented. Medical tasks may need clinical context, chain-of-thought prompts tailored to diagnostic reasoning
- **Scoring/Reward signal**: The SMC particle weighting may need a different reward or verifier signal suited to medical answers rather than math correctness
- **Model choice**: May benefit from medical-domain models (e.g., Med-PaLM, BioMistral, Meditron) instead of general math-tuned models

The core `power_smc.py` sampling algorithm should be largely reusable as-is — the changes are mostly in the evaluation harness and grading pipeline.

## Code Conventions

- Code maps directly to paper equations/theorems — preserve paper references in comments
- Bug fixes are annotated as BUG-A through BUG-E; optimizations as OPT-1 through OPT-4
- Uses `@torch.inference_mode()` for efficiency
- Dataclass `PowerSMCOutput` for structured results
- JSONL output format for evaluation results

## HPC

- **Conda env**: `medProj`
- **Recommended GPU**: H100 PCIe (80GB, supports bfloat16 + flash-attn)
- **V100 fallback**: 32GB, float16 only, no flash-attn, reduce n_particles to 16
- **Attention**: flash_attention_2 → sdpa (auto-fallback if flash-attn not installed)

```bash
# Interactive H100 session
srun --nodes=1 --ntasks-per-node=1 --time=5:00:00 --job-name=SimpleJob \
    --cpus-per-task=4 --mem-per-cpu=8G --partition=normal \
    --gres=gpu:nvidia_h100_pcie:1 --pty bash

# Batch job (full eval)
sbatch run_math500_eval.slurm
```
