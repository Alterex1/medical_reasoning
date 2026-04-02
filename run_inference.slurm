#!/bin/bash
#SBATCH --job-name=qwen_grpo
#SBATCH --mail-user=ji798551@ucf.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --account=am014601.ucf

module restore

hostname;pwd;date

export PYTHONDONTWRITEBYTECODE=1

export CACHE_BASE="/blue/am014601.ucf/ji798551.ucf/cache_dirs"
mkdir -p $CACHE_BASE

export TORCHINDUCTOR_CACHE_DIR="$CACHE_BASE/torch_inductor"
export TRITON_CACHE_DIR="$CACHE_BASE/triton"
export VLLM_CACHE_ROOT="$CACHE_BASE/vllm"
export XDG_CACHE_HOME="$CACHE_BASE/xdg"
export HF_HOME="$CACHE_BASE/huggingface"
export FLASHINFER_CACHE_DIR="$CACHE_BASE/flashinfer"
# export TRANSFORMERS_OFFLINE=1                       # never hit the network
export HF_DATASETS_OFFLINE=1

mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR $VLLM_CACHE_ROOT $XDG_CACHE_HOME $FLASHINFER_CACHE_DIR $HF_HOME

export TMPDIR="$CACHE_BASE/tmp"
mkdir -p $TMPDIR

# This is the big one for your specific error:
export TORCHINDUCTOR_DISABLE_REMOTE_CACHE=1

conda activate jiv_finetune

# python eval_greedy.py \
#     --dataset MATH500.json \
#     --model unsloth/Llama-3.2-3B-Instruct \
#     --max_new_tokens 2048 \
#     --output llama32_3b_instruct_math500_greedy_results.jsonl 

python entropy_script.py \
    --model Qwen/Qwen2.5-7B \
    --dataset MATH500.json \
    --data_text_key prompt \
    --output entropy_qwen25_7b.jsonl

# python eval_temp.py \
#     --dataset MATH500.json \
#     --model Qwen/Qwen2.5-3B \
#     --max_new_tokens 2048 \
#     --n_rollouts 16 \
#     --temperature 1.0 \
#     --data_text_key prompt \
#     --output qwen25_3b_base_math500_base10_results.jsonl 


# python eval_temp.py \
#     --dataset MATH500.json \
#     --model /blue/am014601.ucf/ji798551.ucf/Qwen2.5-3B-Curriculum-GRPO-MATH12k-v2/phase_medium/last \
#     --max_new_tokens 2048 \
#     --n_rollouts 16 \
#     --temperature 1.0 \
#     --data_text_key prompt \
#     --output qwen25_3b_grpo_medium_math500_base10_results.jsonl 

# python eval_temp.py \
#     --dataset MATH500.json \
#     --model /blue/am014601.ucf/ji798551.ucf/Qwen2.5-3B-Curriculum-GRPO-MATH12k-v2/phase_easy/last \
#     --max_new_tokens 2048 \
#     --n_rollouts 16 \
#     --temperature 1.0 \
#     --data_text_key prompt \
#     --output qwen25_3b_grpo_easy_math500_base10_results.jsonl 

# python eval_temp.py \
#     --dataset MATH500.json \
#     --model /blue/am014601.ucf/ji798551.ucf/Qwen2.5-3B-GRPO-MATH12k-Base-Rollouts-v1 \
#     --max_new_tokens 2048 \
#     --n_rollouts 16 \
#     --temperature 1.0 \
#     --data_text_key prompt \
#     --output qwen25_3b_grpo_baseline_math500_base10_results.jsonl 

# python eval_temp.py \
#     --dataset math12k.json \
#     --model Qwen/Qwen2.5-3B \
#     --max_new_tokens 2048 \
#     --n_rollouts 16 \
#     --temperature 1.0 \
#     --data_text_key problem \
#     --start_index 2999 \
#     --max_problems 2999 \
#     --output qwen25_3b_base_math12k_base10_results_p2.jsonl 

# python eval_temp.py \
#     --dataset math12k.json \
#     --model Qwen/Qwen2.5-3B \
#     --max_new_tokens 2048 \
#     --n_rollouts 16 \
#     --temperature 1.0 \
#     --data_text_key problem \
#     --start_index 5998 \
#     --max_problems 2999 \
#     --output qwen25_3b_base_math12k_base10_results_p3.jsonl 

# python eval_temp.py \
#     --dataset math12k.json \
#     --model Qwen/Qwen2.5-3B \
#     --max_new_tokens 2048 \
#     --n_rollouts 16 \
#     --temperature 1.0 \
#     --data_text_key problem \
#     --start_index 8997 \
#     --max_problems 2999 \
#     --output qwen25_3b_base_math12k_base10_results_p4.jsonl 

# python -c "
# import json
# parts = [
#     'qwen25_3b_base_math12k_base10_results_p1.jsonl',
#     'qwen25_3b_base_math12k_base10_results_p2.jsonl',
#     'qwen25_3b_base_math12k_base10_results_p3.jsonl',
#     'qwen25_3b_base_math12k_base10_results_p4.jsonl',
# ]
# with open('qwen25_3b_base_math12k_base10_results.jsonl', 'w') as fout:
#     for path in parts:
#         for line in open(path):
#             rec = json.loads(line)
#             if rec.get('_type') == 'summary': continue
#             fout.write(line)
# print('Done')
# "




# python eval_temp.py \
#     --dataset MATH500.json \
#     --model unsloth/Llama-3.2-3B-Instruct \
#     --max_new_tokens 2048 \
#     --n_rollouts 16 \
#     --temperature 0.25 \
#     --output llama32_3b_instruct_math500_temp025_results.jsonl 

# python eval_power_smc.py \
#     --dataset MATH500.json \
#     --model Qwen/Qwen2.5-3B \
#     --n_particles 8 \
#     --max_new_tokens 2048 \
#     --n_rollouts 16 \
#     --prompt_batch_size 1 \
#     --alpha 4.0 \
#     --start_index 0 \
#     --max_examples 50 \
#     --output qwen25_3b_base_math500_power025_nparticles8_results_p1.jsonl 

# python eval_weight_analysis.py \
#     --dataset      MATH500.json \
#     --model        Qwen/Qwen2.5-3B \
#     --output       weight_analysis.pkl \
#     --n_prompts    100 \
#     --alpha        4.0 \
#     --n_particles  64

# python inference_eval.py \
#     --model_path Qwen/Qwen3-4B \
#     --dataset_name /home/ji798551.ucf/power_rl/math_data/math_500.parquet \
#     --n_particles 64 \
#     --n_rollouts 32 \
#     --prompt_batch_size 4 \
#     --alpha 1.25 \
#     --start_index 0 \
#     --max_examples 250 \
#     --output_path qwen3_4b_base_math500_n200_temp08_power_p1.jsonl

# python math_eval.py \
#     --dataset MATH500.json \
#     --model Qwen/Qwen2.5-7B \
#     --output qwen25_7b_base_math500_n16_base_temp025_greedy_results.jsonl \
#     --n_rollouts 16 \
#     --max_new_tokens 3072 \
#     --temp_temperature 0.25

# python vllm_inference_eval.py \
#     --model_path Qwen/Qwen3-4B \
#     --dataset_name /home/ji798551.ucf/power_rl/math_data/math_500.parquet \
#     --temperature 0.25 \
#     --n_samples 32 \
#     --max_new_tokens 2048 \
#     --output_path qwen3_4b_base_math500_n32_temp025.jsonl

# python per_prompt_eval_math.py \
#     --input qwen3_4b_base_math500_n200_temp08.jsonl \
#     --output qwen3_4b_base_math500_n200_temp08_per_prompt_subset.jsonl \
#     --max_k 32 \
#     --max_bon 32 

# python per_prompt_eval_math.py \
#     --input qwen3_4b_base_math500_n200_temp10.jsonl \
#     --output qwen3_4b_base_math500_n200_temp10_per_prompt_n32_subset.jsonl \
#     --max_k 32 \
#     --max_bon 32
# python per_prompt_eval_math.py \
#     --input qwen3_4b_base_math500_n200_temp025.jsonl \
#     --output qwen3_4b_base_math500_n200_temp025_per_prompt_n32_subset.jsonl \
#     --max_k 32 \
#     --max_bon 32
# python per_prompt_eval_math.py \
#     --input qwen3_4b_base_math500_n200_temp025_power_p1_grouped.jsonl \
#     --output qwen3_4b_base_math500_n200_temp025_power_p1_per_prompt_n32_subset.jsonl \
#     --max_k 32 \
#     --max_bon 32
date