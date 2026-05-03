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

WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu126torch2.10-cp310-cp310-linux_x86_64.whl"

echo "Installing prebuilt flash-attn wheel..."
echo "$WHEEL_URL"
pip install "$WHEEL_URL"

echo "Verifying install..."
python -c "import flash_attn; print('flash-attn version:', flash_attn.__version__)"

echo "Done"
date
