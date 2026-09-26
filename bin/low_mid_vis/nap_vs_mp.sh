#!/bin/bash

MODELS=(
  "resnet101.gluon_in1k"
  "convnext_base.fb_in1k convnext_large.fb_in1k"
  # "resnet50s.gluon_in1k resnext101_32x8d.fb_swsl_ig1b_ft_in1k resnext101_32x4d.fb_swsl_ig1b_ft_in1k"
  # "convnext_tiny.fb_in1k convnext_base.clip_laion2b_augreg_ft_in1k convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384 convnext_xlarge.fb_in22k_ft_in1k"
  # "vit_base_patch16_clip_224.openai_ft_in12k_in1k vit_large_patch14_clip_224.laion2b_ft_in12k_in1k vit_large_patch14_clip_224.openai_ft_in12k_in1k"
  # "deit3_base_patch16_224.fb_in1k deit3_medium_patch16_224.fb_in1k deit3_large_patch16_224.fb_in22k_ft_in1k"
  # "focalnet_base_lrf.ms_in1k focalnet_base_srf.ms_in1k"
  # "swin_base_patch4_window7_224.ms_in1k swin_s3_base_224.ms_in1k swinv2_base_window12to16_192to256.ms_in22k_ft_in1k"
)

RECORD_FROM=(
  "^act1:in ^layer[1-4]\.([0-9]|[1-2][0-9])\.act3:in ^fc:in"
  "^stem\.1 ^stages\.[0-3]\.blocks\.([0-9]|[1-2][0-9]):in"
  "^blocks\.([0-9]|[1-9][0-9]):out"
  "^blocks\.([0-9]|[1-9][0-9]):out"
  "^layers\.[0-3]\.blocks\.([0-9]|1[0-9])"
  "^layers\.[0-3]\.blocks\.([0-9]|[1-3][0-9])"
)

for i in "${!MODELS[@]}"; do
uv run python -m scripts.low_mid_vis.nap_vs_mp \
  --annotations_file "data/datasets/low_mid_vision/nap_vs_mp_3d_geons/annotation.csv" \
  --models ${MODELS[$i]} \
  --record_from ${RECORD_FROM[$i]} \
  --overwrite_recordings \
  --results_folder data/results \
  --output_tag 'pre_act'
done


RECORD_FROM=(
  "^layer[1-4]\.([0-9]|[1-2][0-9])\.bn3"
  "^stages\.[0-3]\.blocks\.([0-9]|[1-2][0-9]).drop_path"
  "^blocks\.([0-9]|[1-9][0-9])\.drop_path2"
  "^blocks\.([0-9]|[1-9][0-9])\.drop_path2"
  "^layers\.[0-3]\.blocks\.([0-9]|1[0-9])\.drop_path2"
  "^layers\.[0-3]\.blocks\.([0-9]|[1-3][0-9])\.drop_path2"
)

for i in "${!MODELS[@]}"; do
uv run python -m scripts.low_mid_vis.nap_vs_mp \
  --annotations_file "data/datasets/low_mid_vision/nap_vs_mp_3d_geons/annotation.csv" \
  --models ${MODELS[$i]} \
  --record_from ${RECORD_FROM[$i]} \
  --overwrite_recordings \
  --results_folder data/results \
  --output_tag 'res_stream'
done
