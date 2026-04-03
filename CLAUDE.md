# CLAUDE.md

## Project Overview

Adapting Power-SMC (Power Sequential Monte Carlo) for **medical reasoning**. Power-SMC is a low-latency, training-free reasoning framework for LLMs from the paper "Power-SMC: Low-Latency Sequence-Level Power Sampling for Training-Free LLM Reasoning" (arXiv:2602.10273).

The existing implementation targets math reasoning (MATH500 benchmark). The goal of this project is to adapt it for medical reasoning tasks — this requires changes to the evaluation pipeline, grading logic, dataset, and potentially the prompting/scoring strategy.

## Repository Structure

```
├── core/                          # Domain-agnostic SMC algorithm
│   └── power_smc.py               #   PowerSMC class
│
├── eval/                          # Evaluation harnesses
│   ├── math/                      #   Math reasoning (MATH500)
│   │   ├── eval_power_smc.py      #     Math eval script
│   │   ├── grader.py              #     SymPy/LaTeX answer grading
│   │   └── math_normalize.py      #     Expression normalization
│   └── medical/                   #   Medical reasoning (VQA-RAD)
│       ├── eval_medical_smc.py    #     Medical VQA Power-SMC eval
│       ├── eval_medical_baseline.py #   Medical VQA baseline eval (greedy/temp)
│       └── medical_grader.py      #     Medical answer grading
│
├── data/                          # Datasets and download scripts
│   ├── MATH500.json               #   500 math problems
│   └── download_medqa.py          #   Downloads VQA-RAD dataset
│
├── scripts/                       # SLURM job scripts
│   ├── run_inference.slurm        #   Legacy inference job
│   ├── run_math500_eval.slurm     #   Full MATH500 eval job
│   ├── run_medical_eval.slurm     #   Full medical VQA eval (SMC + baselines)
│   └── install_flash_attn.slurm   #   CPU-only flash-attn build
│
├── results/                       # Evaluation output (JSONL)
├── requirements.txt               # Pinned Python dependencies
├── CLAUDE.md
└── README.md
```

## Tech Stack

- **Language**: Python 3
- **Core deps**: PyTorch (2.10+cu126), Transformers (5.4), NumPy, SymPy, math-verify, tqdm
- **Other notable deps**: latex2sympy2_extended, pandas, pyarrow, safetensors, rich, triton
- **Models**: HuggingFace pretrained models (e.g., `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-VL-7B-Instruct`)
- **GPU**: CUDA 12.6 (nvidia libs pinned in requirements)

## Install Dependencies

```bash
pip install -r requirements.txt

# Flash Attention 2 (optional, requires Ampere+ GPU i.e. A100/H100/H200)
# Build from source — needs CUDA toolkit and takes 10-20 min with 16 CPUs
# For CPU-only build node, set the target arch explicitly:
TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=8 pip install flash-attn --no-build-isolation

# Or submit as a SLURM job:
sbatch scripts/install_flash_attn.slurm
```

If flash-attn is not installed, the code automatically falls back to PyTorch SDPA (no extra install needed).

## Running

All commands run from the **repo root**.

### Math Evaluation (MATH500)

```bash
# Full eval (H100, bfloat16, flash-attn)
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

# Quick test run (5 problems)
python -m eval.math.eval_power_smc \
    --dataset data/MATH500.json \
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
python -m eval.math.eval_power_smc \
    --dataset data/MATH500.json \
    --model Qwen/Qwen2.5-7B \
    --output results/power_smc.jsonl \
    --dtype float16 \
    --alpha 4.0 \
    --n_particles 16 \
    --n_rollouts 4 \
    --prompt_batch_size 1 \
    --max_new_tokens 512

# Resume interrupted runs
python -m eval.math.eval_power_smc ... --resume
```

### Medical VQA Evaluation (VQA-RAD)

```bash
# Download dataset first
python data/download_medqa.py

# Power-SMC eval (full test set, H100)
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

# Greedy baseline (deterministic, 1 rollout per question)
python -m eval.medical.eval_medical_baseline \
    --dataset data/vqa_rad/VQA_RAD_test.json \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results/medical_baseline_greedy_full.jsonl \
    --dtype bfloat16 \
    --temperature 0.0 \
    --max_new_tokens 512

# Temperature baseline (stochastic, 4 rollouts for pass@k)
python -m eval.medical.eval_medical_baseline \
    --dataset data/vqa_rad/VQA_RAD_test.json \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results/medical_baseline_temp_full.jsonl \
    --dtype bfloat16 \
    --temperature 1.0 \
    --n_rollouts 4 \
    --max_new_tokens 512

# Quick test run (5 questions)
python -m eval.medical.eval_medical_smc ... --max_examples 5

# Resume interrupted runs
python -m eval.medical.eval_medical_smc ... --resume

# Batch job (runs all three evals sequentially on H100)
sbatch scripts/run_medical_eval.slurm
```

## Key Parameters

- `--alpha` (default=4.0) — Power exponent for importance distribution
- `--n_particles` (default=64) — Number of SMC particles/samples
- `--kappa` (default=0.5) — ESS resampling threshold
- `--resample` ("systematic"|"multinomial") — Resampling method
- `--max_new_tokens` (default=2048) — Max generation length
- `--n_rollouts` (default=1) — Independent SMC runs per problem

## Medical Reasoning Adaptation

The medical VQA pipeline adapts Power-SMC for visual medical question answering. The core SMC algorithm (`core/power_smc.py`) is reused unchanged — only the eval harness and grading were adapted.

### What changed from math:
- **Dataset**: VQA-RAD (314 radiology images, 451 test QA pairs) instead of MATH500
- **Model**: Qwen2.5-VL-7B-Instruct (vision-language) instead of Qwen2.5-7B (text-only)
- **VLM support**: `power_smc.py` accepts `prefill_kwargs` to pass image data (pixel_values, image_grid_thw) during prefill only; decode loop is unchanged
- **Grading**: String-based matching with medical normalization instead of SymPy/LaTeX
- **Prompts**: Separate CoT prompts for closed-ended (yes/no) vs open-ended questions
- **Baseline comparison**: `eval_medical_baseline.py` provides greedy and temperature sampling baselines using standard `model.generate()` for direct comparison against Power-SMC

## Code Conventions

- Code maps directly to paper equations/theorems — preserve paper references in comments
- Bug fixes are annotated as BUG-A through BUG-E; optimizations as OPT-1 through OPT-4
- Uses `@torch.inference_mode()` for efficiency
- Dataclass `PowerSMCOutput` for structured results
- JSONL output format for evaluation results

## HPC

- **Conda env**: Create your own (e.g., `conda create -n myenv python=3.10`)
- **Recommended GPU**: H100 PCIe (80GB, supports bfloat16 + flash-attn)
- **V100 fallback**: 32GB, float16 only, no flash-attn, reduce n_particles to 16
- **Attention**: flash_attention_2 → sdpa (auto-fallback if flash-attn not installed)

```bash
# Interactive H100 session
srun --nodes=1 --ntasks-per-node=1 --time=5:00:00 --job-name=SimpleJob \
    --cpus-per-task=4 --mem-per-cpu=8G --partition=normal \
    --gres=gpu:nvidia_h100_pcie:1 --pty bash

# Batch jobs
sbatch scripts/run_math500_eval.slurm    # Math eval
sbatch scripts/run_medical_eval.slurm    # Medical VQA eval (all 3 methods)
```
