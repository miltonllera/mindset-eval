#!/bin/bash
# ==============================================================================
# Automatically generated chunked execution script
# Target model:        focalnet_base_lrf.ms_in1k
# Experiment script:   scripts.low_mid_vis.crowding
# Filter patterns:     Conv2d, Linear
# Total matched layers: 197
# Total chunks:        5
# ==============================================================================
set -euo pipefail

MODELS=(
  focalnet_base_lrf.ms_in1k
)

ANNOTATIONS_FILE="data/datasets/low_mid_vision/un_crowding/annotation.csv"

# ------------------------------------------------------------------------------
# Chunk 1/5: 40 layers (0: Conv2d -> 119: Conv2d)
# ------------------------------------------------------------------------------
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Chunk 1/5 (chunk_0)..."
uv run python -m scripts.low_mid_vis.crowding \
  --annotations_file "${ANNOTATIONS_FILE}" \
  --models "${MODELS[@]}" \
  --output_tag "chunk_0" \
  --overwrite_recordings \
  --record_from \
    "^0: Conv2d$" \
    "^4: Conv2d$" \
    "^5: Conv2d$" \
    "^7: Conv2d$" \
    "^9: Conv2d$" \
    "^11: Conv2d$" \
    "^13: Conv2d$" \
    "^20: Conv2d$" \
    "^24: Conv2d$" \
    "^30: Conv2d$" \
    "^31: Conv2d$" \
    "^33: Conv2d$" \
    "^35: Conv2d$" \
    "^37: Conv2d$" \
    "^39: Conv2d$" \
    "^46: Conv2d$" \
    "^50: Conv2d$" \
    "^55: Conv2d$" \
    "^58: Conv2d$" \
    "^59: Conv2d$" \
    "^61: Conv2d$" \
    "^63: Conv2d$" \
    "^65: Conv2d$" \
    "^67: Conv2d$" \
    "^74: Conv2d$" \
    "^78: Conv2d$" \
    "^84: Conv2d$" \
    "^85: Conv2d$" \
    "^87: Conv2d$" \
    "^89: Conv2d$" \
    "^91: Conv2d$" \
    "^93: Conv2d$" \
    "^100: Conv2d$" \
    "^104: Conv2d$" \
    "^109: Conv2d$" \
    "^112: Conv2d$" \
    "^113: Conv2d$" \
    "^115: Conv2d$" \
    "^117: Conv2d$" \
    "^119: Conv2d$"

# ------------------------------------------------------------------------------
# Chunk 2/5: 40 layers (121: Conv2d -> 249: Conv2d)
# ------------------------------------------------------------------------------
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Chunk 2/5 (chunk_1)..."
uv run python -m scripts.low_mid_vis.crowding \
  --annotations_file "${ANNOTATIONS_FILE}" \
  --models "${MODELS[@]}" \
  --output_tag "chunk_1" \
  --overwrite_recordings \
  --record_from \
    "^121: Conv2d$" \
    "^128: Conv2d$" \
    "^132: Conv2d$" \
    "^138: Conv2d$" \
    "^139: Conv2d$" \
    "^141: Conv2d$" \
    "^143: Conv2d$" \
    "^145: Conv2d$" \
    "^147: Conv2d$" \
    "^154: Conv2d$" \
    "^158: Conv2d$" \
    "^164: Conv2d$" \
    "^165: Conv2d$" \
    "^167: Conv2d$" \
    "^169: Conv2d$" \
    "^171: Conv2d$" \
    "^173: Conv2d$" \
    "^180: Conv2d$" \
    "^184: Conv2d$" \
    "^190: Conv2d$" \
    "^191: Conv2d$" \
    "^193: Conv2d$" \
    "^195: Conv2d$" \
    "^197: Conv2d$" \
    "^199: Conv2d$" \
    "^206: Conv2d$" \
    "^210: Conv2d$" \
    "^216: Conv2d$" \
    "^217: Conv2d$" \
    "^219: Conv2d$" \
    "^221: Conv2d$" \
    "^223: Conv2d$" \
    "^225: Conv2d$" \
    "^232: Conv2d$" \
    "^236: Conv2d$" \
    "^242: Conv2d$" \
    "^243: Conv2d$" \
    "^245: Conv2d$" \
    "^247: Conv2d$" \
    "^249: Conv2d$"

# ------------------------------------------------------------------------------
# Chunk 3/5: 40 layers (251: Conv2d -> 379: Conv2d)
# ------------------------------------------------------------------------------
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Chunk 3/5 (chunk_2)..."
uv run python -m scripts.low_mid_vis.crowding \
  --annotations_file "${ANNOTATIONS_FILE}" \
  --models "${MODELS[@]}" \
  --output_tag "chunk_2" \
  --overwrite_recordings \
  --record_from \
    "^251: Conv2d$" \
    "^258: Conv2d$" \
    "^262: Conv2d$" \
    "^268: Conv2d$" \
    "^269: Conv2d$" \
    "^271: Conv2d$" \
    "^273: Conv2d$" \
    "^275: Conv2d$" \
    "^277: Conv2d$" \
    "^284: Conv2d$" \
    "^288: Conv2d$" \
    "^294: Conv2d$" \
    "^295: Conv2d$" \
    "^297: Conv2d$" \
    "^299: Conv2d$" \
    "^301: Conv2d$" \
    "^303: Conv2d$" \
    "^310: Conv2d$" \
    "^314: Conv2d$" \
    "^320: Conv2d$" \
    "^321: Conv2d$" \
    "^323: Conv2d$" \
    "^325: Conv2d$" \
    "^327: Conv2d$" \
    "^329: Conv2d$" \
    "^336: Conv2d$" \
    "^340: Conv2d$" \
    "^346: Conv2d$" \
    "^347: Conv2d$" \
    "^349: Conv2d$" \
    "^351: Conv2d$" \
    "^353: Conv2d$" \
    "^355: Conv2d$" \
    "^362: Conv2d$" \
    "^366: Conv2d$" \
    "^372: Conv2d$" \
    "^373: Conv2d$" \
    "^375: Conv2d$" \
    "^377: Conv2d$" \
    "^379: Conv2d$"

# ------------------------------------------------------------------------------
# Chunk 4/5: 40 layers (381: Conv2d -> 509: Conv2d)
# ------------------------------------------------------------------------------
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Chunk 4/5 (chunk_3)..."
uv run python -m scripts.low_mid_vis.crowding \
  --annotations_file "${ANNOTATIONS_FILE}" \
  --models "${MODELS[@]}" \
  --output_tag "chunk_3" \
  --overwrite_recordings \
  --record_from \
    "^381: Conv2d$" \
    "^388: Conv2d$" \
    "^392: Conv2d$" \
    "^398: Conv2d$" \
    "^399: Conv2d$" \
    "^401: Conv2d$" \
    "^403: Conv2d$" \
    "^405: Conv2d$" \
    "^407: Conv2d$" \
    "^414: Conv2d$" \
    "^418: Conv2d$" \
    "^424: Conv2d$" \
    "^425: Conv2d$" \
    "^427: Conv2d$" \
    "^429: Conv2d$" \
    "^431: Conv2d$" \
    "^433: Conv2d$" \
    "^440: Conv2d$" \
    "^444: Conv2d$" \
    "^450: Conv2d$" \
    "^451: Conv2d$" \
    "^453: Conv2d$" \
    "^455: Conv2d$" \
    "^457: Conv2d$" \
    "^459: Conv2d$" \
    "^466: Conv2d$" \
    "^470: Conv2d$" \
    "^476: Conv2d$" \
    "^477: Conv2d$" \
    "^479: Conv2d$" \
    "^481: Conv2d$" \
    "^483: Conv2d$" \
    "^485: Conv2d$" \
    "^492: Conv2d$" \
    "^496: Conv2d$" \
    "^502: Conv2d$" \
    "^503: Conv2d$" \
    "^505: Conv2d$" \
    "^507: Conv2d$" \
    "^509: Conv2d$"

# ------------------------------------------------------------------------------
# Chunk 5/5: 37 layers (511: Conv2d -> 637: Linear)
# ------------------------------------------------------------------------------
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting Chunk 5/5 (chunk_4)..."
uv run python -m scripts.low_mid_vis.crowding \
  --annotations_file "${ANNOTATIONS_FILE}" \
  --models "${MODELS[@]}" \
  --output_tag "chunk_4" \
  --overwrite_recordings \
  --record_from \
    "^511: Conv2d$" \
    "^518: Conv2d$" \
    "^522: Conv2d$" \
    "^528: Conv2d$" \
    "^529: Conv2d$" \
    "^531: Conv2d$" \
    "^533: Conv2d$" \
    "^535: Conv2d$" \
    "^537: Conv2d$" \
    "^544: Conv2d$" \
    "^548: Conv2d$" \
    "^554: Conv2d$" \
    "^555: Conv2d$" \
    "^557: Conv2d$" \
    "^559: Conv2d$" \
    "^561: Conv2d$" \
    "^563: Conv2d$" \
    "^570: Conv2d$" \
    "^574: Conv2d$" \
    "^579: Conv2d$" \
    "^582: Conv2d$" \
    "^583: Conv2d$" \
    "^585: Conv2d$" \
    "^587: Conv2d$" \
    "^589: Conv2d$" \
    "^591: Conv2d$" \
    "^598: Conv2d$" \
    "^602: Conv2d$" \
    "^608: Conv2d$" \
    "^609: Conv2d$" \
    "^611: Conv2d$" \
    "^613: Conv2d$" \
    "^615: Conv2d$" \
    "^617: Conv2d$" \
    "^624: Conv2d$" \
    "^628: Conv2d$" \
    "^637: Linear$"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] All chunks completed successfully!"
