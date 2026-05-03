#!/bin/bash
# Optional LLaVA-Med attention patch.
#
# Purpose:
#   LLaVA-Med's loader does not expose an attn_implementation argument, so the
#   adapter cannot choose FlashAttention 2 directly. This script updates the
#   installed LLaVA-Med builder.py in the active conda environment:
#
#     - If flash_attn is installed and the GPU is Ampere+ (A100/H100/H200),
#       inject attn_implementation="flash_attention_2".
#     - Otherwise, remove any FlashAttention-specific kwargs so LLaVA-Med uses
#       the default PyTorch SDPA path.
#
# Safe to re-run. For a full environment reset, rerun scripts/setup_env.sh.

set -e

if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: no conda env activated. Run 'conda activate medProj' first."
    exit 1
fi

BUILDER="$CONDA_PREFIX/lib/python3.10/site-packages/llava/model/builder.py"
if [ ! -f "$BUILDER" ]; then
    echo "ERROR: $BUILDER not found"
    echo "       Install LLaVA-Med first, e.g. bash scripts/setup_env.sh"
    exit 1
fi

set_sdpa_default() {
    echo "Configuring LLaVA-Med for PyTorch SDPA/default attention..."
    sed -i -E 's/[[:space:]]*attn_implementation="flash_attention_2",//g' "$BUILDER"
    sed -i -E 's/[[:space:]]*use_flash_attention_2=(True|False),?//g' "$BUILDER"
    echo "  SDPA/default attention path is active."
}

if ! python -c "import flash_attn" 2>/dev/null; then
    echo "flash_attn is not importable in this environment."
    set_sdpa_default
    exit 0
fi

CAP=$(python -c "import torch; print(torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0)" 2>/dev/null || echo "0")
if [ "$CAP" -lt "8" ]; then
    echo "GPU compute capability is ${CAP}.x; FlashAttention 2 requires Ampere+ (8.0+)."
    set_sdpa_default
    exit 0
fi

# Strip deprecated transformers 4.x kwarg before adding the transformers 5.x
# attention kwarg.
sed -i -E 's/[[:space:]]*use_flash_attention_2=(True|False),?//g' "$BUILDER"

if grep -q 'attn_implementation="flash_attention_2"' "$BUILDER"; then
    echo "LLaVA-Med already uses attn_implementation=\"flash_attention_2\"."
else
    sed -i -E 's/(low_cpu_mem_usage=False,)/\1 attn_implementation="flash_attention_2",/' "$BUILDER"
    echo "Enabled FlashAttention 2 in: $BUILDER"
fi

echo ""
echo "Resulting from_pretrained call in builder.py:"
grep -A1 "low_cpu_mem_usage" "$BUILDER" | head -10
