# Medical Reasoning with Power-SMC

This repository adapts **Power-SMC** (Power Sequential Monte Carlo) from
training-free math reasoning to **medical visual question answering**. The core
sampler is kept domain-agnostic, while the evaluation harness, prompts, model
adapters, datasets, and grading logic are adapted for medical VQA.

The main research question is whether sequence-level power sampling can improve
medical reasoning accuracy over standard greedy or temperature-sampling
baselines without fine-tuning the model.

## Contents

- [Project Status](#project-status)
- [Setup](#setup)
- [Datasets](#datasets)
- [Quick Smoke Tests](#quick-smoke-tests)
- [Inference on New Questions](#inference-on-new-questions)
- [Reproduce Evaluations](#reproduce-evaluations)
- [Inspect Results](#inspect-results)
- [File Structure](#file-structure)
- [Implementation Notes](#implementation-notes)
- [External Code and Attribution](#external-code-and-attribution)

## Project Status

This project is primarily a **medical VQA evaluation pipeline** for applying
Power-SMC to image-based medical reasoning. The original math evaluation path
is still included as a reference implementation and can still be run on
MATH500, but it is not the main focus of this adaptation.

- **Training:** not applicable. Power-SMC is an inference-time, training-free
  decoding method.
- **Primary evaluation:** medical visual question answering on VQA-RAD and
  PMC-VQA.
- **Reference evaluation:** MATH500 support is retained from the original
  math-reasoning setting.
- **Inference/output:** evaluation scripts run model inference and write JSONL
  records with per-sample completions, parsed answers, correctness, pass@k,
  and summary metrics.
- **Models tested:** Qwen2.5-VL, LLaVA-Med, and MedGemma for medical VQA;
  Qwen2.5 text models for the math reference path.

Existing result artifacts live in `results/`. A narrative snapshot is in
`RESULTS_STATUS.txt`; use `scripts/inspect_results.py` to recompute metrics
directly from JSONL files.

## Setup

Run commands from the repository root.

Recommended environment:

- Python 3.10
- CUDA 12.6 capable environment
- H100/A100/H200 for full bfloat16 runs
- V100 32GB fallback with reduced particles/rollouts and `float16`

Create and activate an environment:

```bash
conda create -n medProj python=3.10 -y
conda activate medProj
```

Install pinned dependencies:

```bash
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126
```

For the full medical model suite, especially LLaVA-Med and MedGemma, the
repository includes an HPC-oriented setup script:

```bash
bash scripts/setup_env.sh
```

That script installs the base requirements, updates `transformers` for
MedGemma support, installs LLaVA-Med without downgrading dependencies, and
patches LLaVA-Med's builder for the expected local environment.

Optional FlashAttention 2:

```bash
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu126torch2.10-cp310-cp310-linux_x86_64.whl
```

This pre-built wheel matches CUDA 12.6, PyTorch 2.10, and Python 3.10. If
FlashAttention 2 is unavailable or unsupported, the code falls back to PyTorch
SDPA. Do not use FlashAttention 2 on V100.

Some HuggingFace models are gated or require accepting model terms. If model
loading fails with an authentication error, run:

```bash
huggingface-cli login
```

## Datasets

### VQA-RAD

VQA-RAD is the default radiology VQA benchmark used by the medical evaluation
scripts.

```bash
python data/download_medqa.py
```

This creates:

```text
data/vqa_rad/VQA_RAD_train.json
data/vqa_rad/VQA_RAD_test.json
data/vqa_rad/images/
```

### PMC-VQA

PMC-VQA is a multiple-choice medical VQA benchmark built from PubMed Central
figures. The default command downloads the smaller curated `test_clean` split
from v1.

```bash
python data/download_pmc_vqa.py
```

Other useful variants:

```bash
# Larger v1 test split
python data/download_pmc_vqa.py --split test

# v2 release with a smaller image archive
python data/download_pmc_vqa.py --version 2 --split test

# Metadata only, without image extraction
python data/download_pmc_vqa.py --no-extract
```

Downloaded datasets and images are ignored by git and should be regenerated
with these scripts.

## Quick Smoke Tests

Use a tiny run before submitting a long HPC job.

VQA-RAD Power-SMC smoke test:

```bash
python -m eval.medical.eval_medical_smc \
    --dataset data/vqa_rad/VQA_RAD_test.json \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results/smoke_qwen_smc.jsonl \
    --dtype bfloat16 \
    --alpha 4.0 \
    --n_particles 4 \
    --n_rollouts 1 \
    --prompt_batch_size 1 \
    --max_new_tokens 128 \
    --max_examples 5
```

Greedy baseline smoke test:

```bash
python -m eval.medical.eval_medical_baseline \
    --dataset data/vqa_rad/VQA_RAD_test.json \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results/smoke_qwen_greedy.jsonl \
    --dtype bfloat16 \
    --temperature 0.0 \
    --max_new_tokens 128 \
    --max_examples 5
```

Inspect smoke output:

```bash
python scripts/inspect_results.py results/smoke_qwen_smc.jsonl --smoke
```

## Inference on New Questions

The reproducible inference entrypoint is the medical evaluation CLI. It runs
the model, stores completions, parses answers, and optionally grades them if
ground-truth labels are present.

Create a JSON list with this schema:

```json
[
  {
    "idx": 0,
    "image": "path/to/image.jpg",
    "question": "Is there evidence of cardiomegaly?",
    "answer": "yes",
    "question_type": "closed"
  }
]
```

For PMC-style multiple choice, include `choices` and set `question_type` to
`mcq`:

```json
[
  {
    "idx": 0,
    "image": "path/to/image.jpg",
    "question": "What modality is shown?\nA) CT\nB) MRI\nC) X-ray\nD) Ultrasound",
    "answer": "C",
    "choices": {"A": "CT", "B": "MRI", "C": "X-ray", "D": "Ultrasound"},
    "question_type": "mcq"
  }
]
```

Run Power-SMC inference/evaluation:

```bash
python -m eval.medical.eval_medical_smc \
    --dataset data/custom_medical_vqa.json \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results/custom_power_smc.jsonl \
    --dtype bfloat16 \
    --alpha 4.0 \
    --n_particles 16 \
    --n_rollouts 4 \
    --prompt_batch_size 1 \
    --max_new_tokens 512
```

For unlabeled inference, use the same schema with a placeholder `answer` and
read the generated `completion` fields in the JSONL output. Accuracy fields
will not be meaningful without labels.

## Reproduce Evaluations

### VQA-RAD: Power-SMC

Full Qwen2.5-VL Power-SMC evaluation:

```bash
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
```

Resume an interrupted run:

```bash
python -m eval.medical.eval_medical_smc \
    --dataset data/vqa_rad/VQA_RAD_test.json \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results/medical_power_smc_full.jsonl \
    --resume
```

### VQA-RAD: Baselines

Greedy baseline:

```bash
python -m eval.medical.eval_medical_baseline \
    --dataset data/vqa_rad/VQA_RAD_test.json \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results/medical_baseline_greedy_full.jsonl \
    --dtype bfloat16 \
    --temperature 0.0 \
    --max_new_tokens 512
```

Temperature baseline with 4 rollouts:

```bash
python -m eval.medical.eval_medical_baseline \
    --dataset data/vqa_rad/VQA_RAD_test.json \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results/medical_baseline_temp_full.jsonl \
    --dtype bfloat16 \
    --temperature 1.0 \
    --n_rollouts 4 \
    --max_new_tokens 512
```

Batch job that runs Power-SMC, greedy, and temperature baselines:

```bash
sbatch scripts/run_medical_eval.slurm
```

### PMC-VQA

Power-SMC on the default PMC-VQA `test_clean` split:

```bash
python -m eval.medical.eval_medical_smc \
    --dataset data/pmc_vqa/PMC_VQA_test_clean.json \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results/pmc_vqa_qwen_power_smc_cot.jsonl \
    --dtype bfloat16 \
    --alpha 4.0 \
    --n_particles 16 \
    --n_rollouts 4 \
    --prompt_batch_size 1 \
    --max_new_tokens 512
```

HF baseline:

```bash
python -m eval.medical.eval_medical_baseline \
    --dataset data/pmc_vqa/PMC_VQA_test_clean.json \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --output results/pmc_vqa_qwen_baseline_greedy_cot.jsonl \
    --dtype bfloat16 \
    --temperature 0.0 \
    --max_new_tokens 512
```

PMC-VQA SLURM entrypoints are in `scripts/run_pmc_vqa_*.slurm`.

### MATH500

The math harness is retained as a reference for the original task setting.

```bash
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
```

Quick math run:

```bash
python -m eval.math.eval_power_smc \
    --dataset data/MATH500.json \
    --model Qwen/Qwen2.5-7B \
    --output results/power_smc_smoke.jsonl \
    --dtype bfloat16 \
    --alpha 4.0 \
    --n_particles 16 \
    --n_rollouts 4 \
    --prompt_batch_size 1 \
    --max_new_tokens 512 \
    --max_examples 5
```

## Inspect Results

Each JSONL output has one record per question plus a final summary record.
Per-question records include:

- `idx`, `question`, `image`, `ground_truth`, and `question_type`
- method block: `power_smc` or `baseline`
- per-rollout completions, parsed answer, correctness, token count, finish
  reason, and pass@k
- Power-SMC diagnostics such as SMC log weight, normalized weight, and resample
  count

Compare multiple files:

```bash
python scripts/inspect_results.py \
    results/medical_power_smc_full.jsonl \
    results/medical_baseline_greedy_full.jsonl \
    results/medical_baseline_temp_full.jsonl \
    --compact
```

Show representative completions:

```bash
python scripts/inspect_results.py results/medical_power_smc_full.jsonl --samples 3
```

Re-grade old JSONLs after changing parser or grader logic:

```bash
python scripts/regrade_results.py \
    results/pmc_vqa_qwen_power_smc_cot_v100.jsonl \
    --out results/pmc_vqa_qwen_power_smc_cot_v100_regraded.jsonl
```

## File Structure

```text
core/
  power_smc.py                  Core Power-SMC sampler and utilities

data/
  MATH500.json                  Math benchmark used by reference eval
  download_medqa.py             Downloads VQA-RAD and saves JSON/images
  download_pmc_vqa.py           Downloads PMC-VQA metadata/images

eval/
  math/
    eval_power_smc.py           MATH500 Power-SMC evaluation
    grader.py                   Math answer extraction and grading
    math_normalize.py           Math expression normalization helpers
  medical/
    eval_medical_smc.py         Medical VQA Power-SMC evaluation
    eval_medical_baseline.py    HF greedy/temperature baseline
    medical_grader.py           Medical answer parsing and grading
    model_adapters.py           Qwen, LLaVA-Med, and MedGemma adapters

scripts/
  setup_env.sh                  HPC environment setup helper
  inspect_results.py            Summarize JSONL result files
  regrade_results.py            Re-run current grader on stored outputs
  run_*.slurm                   Batch jobs for math/medical evaluations

results/
  *.jsonl                       Evaluation outputs

logs/
  *.out, *.err                  SLURM logs, ignored by git
```

## Implementation Notes

### Power-SMC

`core/power_smc.py` implements sequence-level power sampling:

```text
pi_alpha(y | x) proportional to p_theta(y | x)^alpha
```

The sampler maintains particles, updates sequence-level importance weights,
tracks effective sample size, and resamples when ESS falls below
`kappa * n_particles`. The code preserves comments that map implementation
blocks to the Power-SMC paper equations, theorem statements, algorithm lines,
and appendix details.

Important parameters:

- `--alpha`: power exponent; default `4.0`
- `--n_particles`: number of SMC particles; default `64`
- `--kappa`: ESS resampling threshold; default `0.5`
- `--resample`: `systematic` or `multinomial`
- `--n_rollouts`: independent completions per problem/question
- `--prompt_batch_size`: independent SMC runs processed in parallel
- `--max_new_tokens`: generation budget

### Medical VQA Adaptation

The medical pipeline keeps the core sampler unchanged in spirit and adapts the
surrounding evaluation components:

- VLM adapters construct model-specific image/text inputs.
- `prefill_kwargs` pass image tensors only during the prefill stage.
- The decode loop remains token-only after the multimodal prefill.
- Prompts are specialized for closed yes/no, open-ended, and MCQ questions.
- Medical grading uses answer normalization and domain-aware string matching
  instead of SymPy/LaTeX verification.
- Baseline scripts use the same prompts and graders as Power-SMC for direct
  comparison.

## External Code and Attribution

This project reuses public models, datasets, and paper methods. Cite these in
the report as well as this README.

### Borrowed or Adapted

- **Power-SMC method:** adapted from "Power-SMC: Low-Latency Sequence-Level
  Power Sampling for Training-Free LLM Reasoning"
  ([arXiv:2602.10273](https://arxiv.org/abs/2602.10273)). The algorithmic
  structure in `core/power_smc.py` follows the paper and includes paper-linked
  comments. This project modifies the sampler for bug fixes, batched rollouts,
  cache-safe resampling, and multimodal prefill support.
- **VQA-RAD dataset:** downloaded from
  [flaviagiammarino/vqa-rad](https://huggingface.co/datasets/flaviagiammarino/vqa-rad).
  Original dataset paper: Lau et al., "A dataset of clinically generated visual
  questions and answers about radiology images", Scientific Data, 2018.
- **PMC-VQA dataset:** downloaded from
  [xmcmic/PMC-VQA](https://huggingface.co/datasets/xmcmic/PMC-VQA). Original
  paper: Zhang et al., "PMC-VQA: Visual Instruction Tuning for Medical Visual
  Question Answering" ([arXiv:2305.10415](https://arxiv.org/abs/2305.10415)).
- **LLaVA-Med support:** uses the Microsoft LLaVA-Med package from
  [github.com/microsoft/LLaVA-Med](https://github.com/microsoft/LLaVA-Med).
  The adapter in this repo handles prompt formatting, image tokenization, image
  preprocessing, and generation-output differences.
- **Medical system prompt:** the role-setting portion is adapted from ViTAR
  ([arXiv:2510.10052](https://arxiv.org/abs/2510.10052)); this repo removes
  ViTAR's tool/cropping-specific JSON format because the evaluation does not
  use those tools.
- **Math grading:** uses SymPy, `math-verify`, and related normalization
  utilities to compare extracted math answers.

### Original in This Repository

- Medical VQA Power-SMC evaluation harness
- Medical baseline evaluation harnesses
- Model adapter abstraction for Qwen2.5-VL, LLaVA-Med, and MedGemma
- VQA-RAD and PMC-VQA preprocessing scripts
- Medical answer parser/grader
- Result inspection and regrading utilities
- SLURM job scripts for the reported experiments

### Modified Relative to the Original Math Setting

- Dataset changed from MATH500 text problems to medical VQA datasets with
  image paths and question metadata.
- Model family changed from text-only LLMs to vision-language models.
- Prompting changed from math chain-of-thought to medical image QA prompts.
- Grading changed from symbolic math verification to normalized medical answer
  matching and MCQ letter matching.
- Power-SMC prefill supports VLM-specific tensors while preserving the
  autoregressive decode loop.
