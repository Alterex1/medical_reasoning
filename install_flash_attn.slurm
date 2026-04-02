#!/bin/bash
#SBATCH --job-name=install_flash_attn
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --partition=normal

hostname; pwd; date

module restore
module list

conda activate medProj

nvcc --version

echo "Building flash-attn for Hopper (sm_90)..."
TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=8 pip install flash-attn --no-build-isolation

echo "Verifying install..."
python -c "import flash_attn; print('flash-attn version:', flash_attn.__version__)"

echo "Done"
date
