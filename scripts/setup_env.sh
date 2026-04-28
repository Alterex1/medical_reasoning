#!/bin/bash
# Full env setup for the medProj conda env. Captures every lesson learned
# from the env-rebuild slog so the next rebuild is one command.
#
# Prereq: a fresh medProj env exists and is currently activated, e.g.
#   conda create -n medProj python=3.10 -y
#   conda activate medProj
#
# Then run:
#   bash scripts/setup_env.sh
#
# Idempotent — safe to re-run after a partial install.

set -e

# ── 0. Sanity check ──────────────────────────────────────────────────────────
if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: no conda env activated. Run 'conda activate medProj' first."
    exit 1
fi
echo "Installing into env: $CONDA_PREFIX"
echo ""

# ── 1. Base requirements (torch comes from PyTorch's CUDA index) ─────────────
echo "[1/4] pip install -r requirements.txt"
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu126

# ── 2. Bump transformers for Med-Gemma support ───────────────────────────────
# Med-Gemma uses AutoModelForImageTextToText (added in 5.4) and works on 5.5.4.
echo ""
echo "[2/4] pip install -U transformers==5.5.4"
pip install -U transformers==5.5.4

# ── 3. LLaVA-Med (no deps to avoid downgrading transformers to 4.36) ─────────
echo ""
echo "[3/4] pip install --no-deps LLaVA-Med"
pip install --no-deps git+https://github.com/microsoft/LLaVA-Med.git

# ── 4. Patch LLaVA-Med's builder.py for transformers 5.x compatibility ───────
# LLaVA-Med hardcodes `use_flash_attention_2=False` in from_pretrained calls.
# That kwarg was renamed in transformers 5.x and now falls through to the
# model __init__ as an unknown kwarg → TypeError. Strip it out.
BUILDER="$CONDA_PREFIX/lib/python3.10/site-packages/llava/model/builder.py"
echo ""
echo "[4/4] Patching LLaVA-Med builder.py"
if [ -f "$BUILDER" ]; then
    if grep -q "use_flash_attention_2" "$BUILDER"; then
        sed -i -E 's/use_flash_attention_2=(True|False),?//g' "$BUILDER"
        echo "  patched: $BUILDER"
    else
        echo "  already patched (no use_flash_attention_2 found)"
    fi
else
    echo "  WARNING: $BUILDER not found — LLaVA-Med may not be installed correctly"
fi

# ── Verification ─────────────────────────────────────────────────────────────
echo ""
echo "=== Verifying imports ==="
python -c "
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText
from llava.model.builder import load_pretrained_model
import tiktoken
print('  torch       :', torch.__version__, '+', torch.version.cuda)
print('  imports OK')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Optional: install flash-attn (Ampere+ GPUs only — A100/H100/H200, NOT V100)"
echo "  Browse https://github.com/Dao-AILab/flash-attention/releases for a wheel"
echo "  matching your torch (cu126, torch2.10, cp310, abi=$(python -c 'import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)'))"
echo "  pip install --no-build-isolation <wheel-url>"
