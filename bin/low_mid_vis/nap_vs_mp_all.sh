#!/bin/bash

RESNET50="resnet50s.gluon_in1k resnet50x4_clip_gap.openai resnet50x16_clip.openai"
# RESNEXT50="resnet101.gluon_in1k resnet101_clip.openai"
RESNEXT101="resnext101_32x4d.fb_swsl_ig1b_ft_in1k resnext101_32x8d.fb_swsl_ig1b_ft_in1k"
CONVNEXT="convnext_base.clip_laion2b_augreg_ft_in1k convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384 convnext_xlarge.fb_in22k_ft_in1k"
VIT="vit_base_patch16_clip_224.openai_ft_in12k_in1k vit_large_patch14_clip_224.laion2b_ft_in12k_in1k vit_large_patch14_clip_224.openai_ft_in12k"
DEIT="deit3_base_patch16_224.fb_in1k deit3_medium_patch16_224.fb_in1k deit3_large_patch16_224.fb_in22k_ft_in1k"
SWIN="swin_base_patch4_window7_224.ms_in1k swinv2_base_window12_192.ms_in22k swin_s3_base_224.ms_in1k"
FOCALNET="focalnet_base_lrf focalnet_large_fl3.ms_in22k focalnet_xlarge_fl3.ms_in22k"



# NAP_vs_MP_3D_geons_standard
uv run python -m scripts.low_mid_vis.nap_vs_mp_change \
  --annotations "data/datasets/full/low_mid_level_vision/NAP_vs_MP_3D_geons_standard/annotation.csv" \
  --change_type geons_standard \
  --models $RESNET50 $RESNEXT101 $CONVNEXT $VIT $DEIT $SWIN $FOCALNET --overwrite_recordings

# NAP_vs_MP_3D_geons_no_shades
uv run python -m scripts.low_mid_vis.nap_vs_mp_change \
  --annotations "data/datasets/full/low_mid_level_vision/NAP_vs_MP_3D_geons_no_shades/annotation.csv" \
  --change_type geons_no_shades \
  --models $RESNET50 $RESNEXT101 $CONVNEXT $VIT $DEIT $SWIN $FOCALNET --overwrite_recordings

# NAP_vs_MP_2D_lines
uv run python -m scripts.low_mid_vis.nap_vs_mp_change \
  --annotations "data/datasets/full/low_mid_level_vision/NAP_vs_MP_2D_lines/annotation.csv" \
  --change_type 2d_lines \
  --models $RESNET50 $RESNEXT101 $CONVNEXT $VIT $DEIT $SWIN $FOCALNET --overwrite_recordings

#NAP_vs_MP_3D_geons_silhouette
uv run python -m scripts.low_mid_vis.nap_vs_mp_change \
  --annotations "data/datasets/full/low_mid_level_vision/NAP_vs_MP_3D_geons_silhouettes/annotation.csv" \
  --change_type 2d_lines \
  --models $RESNET50 $RESNEXT101 $CONVNEXT $VIT $DEIT $SWIN $FOCALNET --overwrite_recordings
