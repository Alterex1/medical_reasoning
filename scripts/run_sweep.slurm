#!/bin/bash
#SBATCH --job-name=smc_sweep
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

conda activate medProj

# ── Sweep configs ─────────────────────────────────────────────────────────────
# Subset: first 100 questions for faster iteration
N_EXAMPLES=100
DATASET="data/vqa_rad/VQA_RAD_test.json"
MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
DTYPE="bfloat16"
ROLLOUTS=4
MAX_TOKENS=512
BATCH=1

# Alpha sweep (fixed: N=16, kappa=0.5)
for ALPHA in 2.0 4.0 6.0 8.0; do
    OUT="results/sweep/alpha${ALPHA}_n16_k0.5.jsonl"
    echo "=== alpha=${ALPHA} n_particles=16 kappa=0.5 ==="
    date
    python -m eval.medical.eval_medical_smc \
        --dataset $DATASET \
        --model $MODEL \
        --output $OUT \
        --dtype $DTYPE \
        --alpha $ALPHA \
        --n_particles 16 \
        --kappa 0.5 \
        --n_rollouts $ROLLOUTS \
        --prompt_batch_size $BATCH \
        --max_new_tokens $MAX_TOKENS \
        --max_examples $N_EXAMPLES
done

# Particle sweep (fixed: alpha=4.0, kappa=0.5)
for NPART in 32 64; do
    OUT="results/sweep/alpha4.0_n${NPART}_k0.5.jsonl"
    echo "=== alpha=4.0 n_particles=${NPART} kappa=0.5 ==="
    date
    python -m eval.medical.eval_medical_smc \
        --dataset $DATASET \
        --model $MODEL \
        --output $OUT \
        --dtype $DTYPE \
        --alpha 4.0 \
        --n_particles $NPART \
        --kappa 0.5 \
        --n_rollouts $ROLLOUTS \
        --prompt_batch_size $BATCH \
        --max_new_tokens $MAX_TOKENS \
        --max_examples $N_EXAMPLES
done

# Kappa sweep (fixed: alpha=4.0, n_particles=16)
for KAPPA in 0.3 0.7; do
    OUT="results/sweep/alpha4.0_n16_k${KAPPA}.jsonl"
    echo "=== alpha=4.0 n_particles=16 kappa=${KAPPA} ==="
    date
    python -m eval.medical.eval_medical_smc \
        --dataset $DATASET \
        --model $MODEL \
        --output $OUT \
        --dtype $DTYPE \
        --alpha 4.0 \
        --n_particles 16 \
        --kappa $KAPPA \
        --n_rollouts $ROLLOUTS \
        --prompt_batch_size $BATCH \
        --max_new_tokens $MAX_TOKENS \
        --max_examples $N_EXAMPLES
done

echo "=== Sweep complete ==="
date

# ── Print summary ─────────────────────────────────────────────────────────────
echo ""
echo "=== RESULTS SUMMARY ==="
for f in results/sweep/*.jsonl; do
    SUMMARY=$(tail -1 "$f")
    CONFIG=$(basename "$f" .jsonl)
    OVERALL=$(echo "$SUMMARY" | python -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['summary']['power_smc']['overall_accuracy']:.4f}\")" 2>/dev/null)
    CLOSED=$(echo "$SUMMARY" | python -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['summary']['power_smc']['closed_accuracy']:.4f}\")" 2>/dev/null)
    OPEN=$(echo "$SUMMARY" | python -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['summary']['power_smc']['open_accuracy']:.4f}\")" 2>/dev/null)
    echo "$CONFIG: overall=$OVERALL  closed=$CLOSED  open=$OPEN"
done
