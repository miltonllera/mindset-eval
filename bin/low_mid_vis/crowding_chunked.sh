#!/bin/bash
# ==============================================================================
# Automatically generated chunked execution script
# Target model:        convnext_base.clip_laion2b_augreg_ft_in1k
# Experiment script:   scripts.low_mid_vis.crowding
# Filter patterns:     Conv2d, Linear
# Total matched layers: 113
# Total chunks:        5
# ==============================================================================
set -euo pipefail

MODELS=(
  # convnext_base.clip_laion2b_augreg_ft_in1k
  # convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384
  # convnext_xlarge.fb_in22k_ft_in1k
)

ANNOTATIONS_FILE="data/datasets/low_mid_vision/un_crowding/annotation.csv"

# ------------------------------------------------------------------------------
# Chunk 1/5: 23 layers (0: Conv2d -> 69: Linear)
# ------------------------------------------------------------------------------
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Chunk 1/5 (chunk_0)..."
uv run python -m scripts.low_mid_vis.crowding \
  --annotations_file "${ANNOTATIONS_FILE}" \
  --models "${MODELS[@]}" \
  --output_tag "chunk_0" \
  --overwrite_recordings \
  --record_from \
    "^0: Conv2d$" \
    "^3: Conv2d$" \
    "^5: Linear$" \
    "^9: Linear$" \
    "^13: Conv2d$" \
    "^15: Linear$" \
    "^19: Linear$" \
    "^23: Conv2d$" \
    "^25: Linear$" \
    "^29: Linear$" \
    "^34: Conv2d$" \
    "^35: Conv2d$" \
    "^37: Linear$" \
    "^41: Linear$" \
    "^45: Conv2d$" \
    "^47: Linear$" \
    "^51: Linear$" \
    "^55: Conv2d$" \
    "^57: Linear$" \
    "^61: Linear$" \
    "^66: Conv2d$" \
    "^67: Conv2d$" \
    "^69: Linear$"

# ------------------------------------------------------------------------------
# Chunk 2/5: 23 layers (73: Linear -> 147: Conv2d)
# ------------------------------------------------------------------------------
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Chunk 2/5 (chunk_1)..."
uv run python -m scripts.low_mid_vis.crowding \
  --annotations_file "${ANNOTATIONS_FILE}" \
  --models "${MODELS[@]}" \
  --output_tag "chunk_1" \
  --overwrite_recordings \
  --record_from \
    "^73: Linear$" \
    "^77: Conv2d$" \
    "^79: Linear$" \
    "^83: Linear$" \
    "^87: Conv2d$" \
    "^89: Linear$" \
    "^93: Linear$" \
    "^97: Conv2d$" \
    "^99: Linear$" \
    "^103: Linear$" \
    "^107: Conv2d$" \
    "^109: Linear$" \
    "^113: Linear$" \
    "^117: Conv2d$" \
    "^119: Linear$" \
    "^123: Linear$" \
    "^127: Conv2d$" \
    "^129: Linear$" \
    "^133: Linear$" \
    "^137: Conv2d$" \
    "^139: Linear$" \
    "^143: Linear$" \
    "^147: Conv2d$"

# ------------------------------------------------------------------------------
# Chunk 3/5: 23 layers (149: Linear -> 223: Linear)
# ------------------------------------------------------------------------------
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Chunk 3/5 (chunk_2)..."
uv run python -m scripts.low_mid_vis.crowding \
  --annotations_file "${ANNOTATIONS_FILE}" \
  --models "${MODELS[@]}" \
  --output_tag "chunk_2" \
  --overwrite_recordings \
  --record_from \
    "^149: Linear$" \
    "^153: Linear$" \
    "^157: Conv2d$" \
    "^159: Linear$" \
    "^163: Linear$" \
    "^167: Conv2d$" \
    "^169: Linear$" \
    "^173: Linear$" \
    "^177: Conv2d$" \
    "^179: Linear$" \
    "^183: Linear$" \
    "^187: Conv2d$" \
    "^189: Linear$" \
    "^193: Linear$" \
    "^197: Conv2d$" \
    "^199: Linear$" \
    "^203: Linear$" \
    "^207: Conv2d$" \
    "^209: Linear$" \
    "^213: Linear$" \
    "^217: Conv2d$" \
    "^219: Linear$" \
    "^223: Linear$"

# ------------------------------------------------------------------------------
# Chunk 4/5: 23 layers (227: Conv2d -> 299: Linear)
# ------------------------------------------------------------------------------
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Chunk 4/5 (chunk_3)..."
uv run python -m scripts.low_mid_vis.crowding \
  --annotations_file "${ANNOTATIONS_FILE}" \
  --models "${MODELS[@]}" \
  --output_tag "chunk_3" \
  --overwrite_recordings \
  --record_from \
    "^227: Conv2d$" \
    "^229: Linear$" \
    "^233: Linear$" \
    "^237: Conv2d$" \
    "^239: Linear$" \
    "^243: Linear$" \
    "^247: Conv2d$" \
    "^249: Linear$" \
    "^253: Linear$" \
    "^257: Conv2d$" \
    "^259: Linear$" \
    "^263: Linear$" \
    "^267: Conv2d$" \
    "^269: Linear$" \
    "^273: Linear$" \
    "^277: Conv2d$" \
    "^279: Linear$" \
    "^283: Linear$" \
    "^287: Conv2d$" \
    "^289: Linear$" \
    "^293: Linear$" \
    "^297: Conv2d$" \
    "^299: Linear$"

# ------------------------------------------------------------------------------
# Chunk 5/5: 21 layers (303: Linear -> 376: Linear)
# ------------------------------------------------------------------------------
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Chunk 5/5 (chunk_4)..."
uv run python -m scripts.low_mid_vis.crowding \
  --annotations_file "${ANNOTATIONS_FILE}" \
  --models "${MODELS[@]}" \
  --output_tag "chunk_4" \
  --overwrite_recordings \
  --record_from \
    "^303: Linear$" \
    "^307: Conv2d$" \
    "^309: Linear$" \
    "^313: Linear$" \
    "^317: Conv2d$" \
    "^319: Linear$" \
    "^323: Linear$" \
    "^327: Conv2d$" \
    "^329: Linear$" \
    "^333: Linear$" \
    "^338: Conv2d$" \
    "^339: Conv2d$" \
    "^341: Linear$" \
    "^345: Linear$" \
    "^349: Conv2d$" \
    "^351: Linear$" \
    "^355: Linear$" \
    "^359: Conv2d$" \
    "^361: Linear$" \
    "^365: Linear$" \
    "^376: Linear$"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] All chunks completed successfully!"
