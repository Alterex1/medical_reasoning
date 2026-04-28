#!/bin/bash
# Optional H100-only patch: enable flash_attention_2 for LLaVA-Med.
#
# LLaVA-Med's load_pretrained_model has a fixed signature with no
# **kwargs, so we can't pass attn_implementation through the adapter.
# Instead, inject the kwarg directly into builder.py's internal
# from_pretrained call.
#
# WHEN TO RUN:
#   - You're on Ampere+ hardware (A100 / H100 / H200)
#   - You have a flash-attn wheel matching your torch + CUDA + Python + ABI
#     installed, and `python -c "import flash_attn"` works
#
# DO NOT RUN ON V100:
#   Flash-attn 2 doesn't support sm_70. With this patch applied,
#   LLaVA-Med would error at model load on V100 because transformers
#   would try to enable FA2 and the runtime check would fail.
#   On V100, leave LLaVA-Med using its default (SDPA).
#
# Idempotent — safe to re-run. To undo, re-run setup_env.sh.

set -e

if [ -z "$CONDA_PREFIX" ]; then
    echo "ERROR: no conda env activated. Run 'conda activate medProj' first."
    exit 1
fi

BUILDER="$CONDA_PREFIX/lib/python3.10/site-packages/llava/model/builder.py"
if [ ! -f "$BUILDER" ]; then
    echo "ERROR: $BUILDER not found"
    exit 1
fi

# Sanity check: flash_attn must be importable
python -c "import flash_attn" 2>/dev/null || {
    echo "ERROR: flash_attn is not importable in this env."
    echo "       Install a matching wheel before running this patch."
    exit 1
}

# Sanity check: must be on Ampere+ (compute cap >= 8.0)
CAP=$(python -c "import torch; print(torch.cuda.get_device_capability()[0])" 2>/dev/null || echo "0")
if [ "$CAP" -lt "8" ]; then
    echo "ERROR: GPU compute capability is $CAP.x — flash-attn 2 needs Ampere+ (8.0+)."
    echo "       This patch would break LLaVA-Med on this hardware. Aborting."
    exit 1
fi

# Inject attn_implementation="flash_attention_2" into the from_pretrained
# call, right after low_cpu_mem_usage=False. Skip if already present.
if grep -q 'attn_implementation="flash_attention_2"' "$BUILDER"; then
    echo "  already patched (attn_implementation found in builder.py)"
else
    sed -i -E 's/(low_cpu_mem_usage=False,)/\1 attn_implementation="flash_attention_2",/' \
        "$BUILDER"
    echo "  patched: $BUILDER (injected attn_implementation=\"flash_attention_2\")"
fi

# Verify
echo ""
echo "Resulting from_pretrained call in builder.py:"
grep -A1 "low_cpu_mem_usage" "$BUILDER" | head -10
