#!/bin/bash

# Available datasets:
# "data/datasets/low_mid_level_vision/NAP_vs_MP_2D_lines/annotation.csv"
# "data/datasets/low_mid_level_vision/NAP_vs_MP_3D_geons_no_shades/annotation.csv"
# "data/datasets/low_mid_level_vision/NAP_vs_MP_3D_geons_silhouettes/annotation.csv"
# "data/datasets/low_mid_level_vision/NAP_vs_MP_3D_geons_standard/annotation.csv"

MODELS=(
  # Swin
  # swin_base_patch4_window7_224.ms_in1k
  # swin_s3_base_224.ms_in1k
  # swinv2_base_window12to16_192to256.ms_in22k_ft_in1k
)

uv run python -m scripts.low_mid_vis.nap_vs_mp \
  --annotations_file "data/datasets/low_mid_vision/nap_vs_mp_3d_geons/annotation.csv" \
  --models "${MODELS[@]}" \
  --overwrite_recordings
