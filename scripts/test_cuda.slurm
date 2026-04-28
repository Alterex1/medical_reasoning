#!/bin/bash
#SBATCH --job-name=test_cuda
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --mail-user=erick.castillo@ucf.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Diagnostic-only: verify the conda env can actually use CUDA on an H100
# PCIe node when launched via sbatch (i.e. with the same shell init path
# the real eval jobs use). Prints every relevant fact, then attempts an
# actual GPU allocation + matmul. Exits 0 if CUDA works, 1 otherwise.

START_TS=$(date +%s)
echo "=== Job started: $(date) ==="
hostname; pwd

# ── Environment ───────────────────────────────────────────────────────────────
module restore
echo ""
echo "--- module list ---"
module list

conda activate medProj

echo ""
echo "--- shell env ---"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "SLURM_STEP_GPUS=$SLURM_STEP_GPUS"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo ""
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv

echo ""
echo "--- python / torch ---"
python -c "
import sys, torch
print('python      :', sys.version.split()[0])
print('torch       :', torch.__version__)
print('torch.cuda  :', torch.version.cuda)
print('cuda avail? :', torch.cuda.is_available())
print('device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device name :', torch.cuda.get_device_name(0))
    print('compute cap :', torch.cuda.get_device_capability(0))
    # Real allocation + tiny matmul to prove it's not lying
    a = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
    c = (a @ b).sum().item()
    print('matmul OK   :', c)
else:
    print('CUDA NOT AVAILABLE — env or cluster is broken')
    sys.exit(1)
"

RC=$?
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
echo ""
echo "=== Job ended: $(date) ==="
printf "=== Total runtime: %dh %dm %ds ===\n" \
    $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60)) $((ELAPSED % 60))
echo "=== Exit code: $RC ==="
exit $RC
