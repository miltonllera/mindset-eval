#!/bin/bash
# ==============================================================================
# Automatically generated chunked execution script
# Target model:        vit_large_patch14_clip_224.openai_ft_in12k_in1k
# Experiment script:   scripts.low_mid_vis.crowding
# Filter patterns:     Conv2d, Linear
# Total matched layers: 98
# Total chunks:        5
# ==============================================================================
set -euo pipefail

MODELS=(
  vit_large_patch14_clip_224.openai_ft_in12k_in1k
  vit_large_patch14_clip_224.laion2b_ft_in12k_in1k
)

ANNOTATIONS_FILE="data/datasets/low_mid_vision/un_crowding/annotation.csv"

# ------------------------------------------------------------------------------
# Chunk 1/5: 20 layers (0: Conv2d -> 92: Linear)
# ------------------------------------------------------------------------------
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Chunk 1/5 (chunk_0)..."
uv run python -m scripts.low_mid_vis.crowding \
  --annotations_file "${ANNOTATIONS_FILE}" \
  --models "${MODELS[@]}" \
  --output_tag "chunk_0" \
  --overwrite_recordings \
  --record_from \
    "^0: Conv2d$" \
    "^6: Linear$" \
    "^11: Linear$" \
    "^16: Linear$" \
    "^20: Linear$" \
    "^25: Linear$" \
    "^30: Linear$" \
    "^35: Linear$" \
    "^39: Linear$" \
    "^44: Linear$" \
    "^49: Linear$" \
    "^54: Linear$" \
    "^58: Linear$" \
    "^63: Linear$" \
    "^68: Linear$" \
    "^73: Linear$" \
    "^77: Linear$" \
    "^82: Linear$" \
    "^87: Linear$" \
    "^92: Linear$"

# ------------------------------------------------------------------------------
# Chunk 2/5: 20 layers (96: Linear -> 187: Linear)
# ------------------------------------------------------------------------------
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Chunk 2/5 (chunk_1)..."
uv run python -m scripts.low_mid_vis.crowding \
  --annotations_file "${ANNOTATIONS_FILE}" \
  --models "${MODELS[@]}" \
  --output_tag "chunk_1" \
  --overwrite_recordings \
  --record_from \
    "^96: Linear$" \
    "^101: Linear$" \
    "^106: Linear$" \
    "^111: Linear$" \
    "^115: Linear$" \
    "^120: Linear$" \
    "^125: Linear$" \
    "^130: Linear$" \
    "^134: Linear$" \
    "^139: Linear$" \
    "^144: Linear$" \
    "^149: Linear$" \
    "^153: Linear$" \
    "^158: Linear$" \
    "^163: Linear$" \
    "^168: Linear$" \
    "^172: Linear$" \
    "^177: Linear$" \
    "^182: Linear$" \
    "^187: Linear$"

# ------------------------------------------------------------------------------
# Chunk 3/5: 20 layers (191: Linear -> 282: Linear)
# ------------------------------------------------------------------------------
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Chunk 3/5 (chunk_2)..."
uv run python -m scripts.low_mid_vis.crowding \
  --annotations_file "${ANNOTATIONS_FILE}" \
  --models "${MODELS[@]}" \
  --output_tag "chunk_2" \
  --overwrite_recordings \
  --record_from \
    "^191: Linear$" \
    "^196: Linear$" \
    "^201: Linear$" \
    "^206: Linear$" \
    "^210: Linear$" \
    "^215: Linear$" \
    "^220: Linear$" \
    "^225: Linear$" \
    "^229: Linear$" \
    "^234: Linear$" \
    "^239: Linear$" \
    "^244: Linear$" \
    "^248: Linear$" \
    "^253: Linear$" \
    "^258: Linear$" \
    "^263: Linear$" \
    "^267: Linear$" \
    "^272: Linear$" \
    "^277: Linear$" \
    "^282: Linear$"

# ------------------------------------------------------------------------------
# Chunk 4/5: 20 layers (286: Linear -> 377: Linear)
# ------------------------------------------------------------------------------
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Chunk 4/5 (chunk_3)..."
uv run python -m scripts.low_mid_vis.crowding \
  --annotations_file "${ANNOTATIONS_FILE}" \
  --models "${MODELS[@]}" \
  --output_tag "chunk_3" \
  --overwrite_recordings \
  --record_from \
    "^286: Linear$" \
    "^291: Linear$" \
    "^296: Linear$" \
    "^301: Linear$" \
    "^305: Linear$" \
    "^310: Linear$" \
    "^315: Linear$" \
    "^320: Linear$" \
    "^324: Linear$" \
    "^329: Linear$" \
    "^334: Linear$" \
    "^339: Linear$" \
    "^343: Linear$" \
    "^348: Linear$" \
    "^353: Linear$" \
    "^358: Linear$" \
    "^362: Linear$" \
    "^367: Linear$" \
    "^372: Linear$" \
    "^377: Linear$"

# ------------------------------------------------------------------------------
# Chunk 5/5: 18 layers (381: Linear -> 464: Linear)
# ------------------------------------------------------------------------------
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Chunk 5/5 (chunk_4)..."
uv run python -m scripts.low_mid_vis.crowding \
  --annotations_file "${ANNOTATIONS_FILE}" \
  --models "${MODELS[@]}" \
  --output_tag "chunk_4" \
  --overwrite_recordings \
  --record_from \
    "^381: Linear$" \
    "^386: Linear$" \
    "^391: Linear$" \
    "^396: Linear$" \
    "^400: Linear$" \
    "^405: Linear$" \
    "^410: Linear$" \
    "^415: Linear$" \
    "^419: Linear$" \
    "^424: Linear$" \
    "^429: Linear$" \
    "^434: Linear$" \
    "^438: Linear$" \
    "^443: Linear$" \
    "^448: Linear$" \
    "^453: Linear$" \
    "^457: Linear$" \
    "^464: Linear$"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] All chunks completed successfully!"
