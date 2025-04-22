#!/bin/bash

# RESNET50="resnet50s.gluon_in1k resnet50x4_clip_gap.openai resnet50x16_clip.openai"
# # RESNEXT50="resnet101.gluon_in1k resnet101_clip.openai"
# RESNEXT101="resnext101_32x4d.fb_swsl_ig1b_ft_in1k resnext101_32x8d.fb_swsl_ig1b_ft_in1k"
# CONVNEXT="convnext_base.clip_laion2b_augreg_ft_in1k convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384 convnext_xlarge.fb_in22k_ft_in1k"
# VIT="vit_base_patch16_clip_224.openai_ft_in12k_in1k vit_large_patch14_clip_224.laion2b_ft_in12k_in1k vit_large_patch14_clip_224.openai_ft_in12k"
# DEIT="deit3_base_patch16_224.fb_in1k deit3_medium_patch16_224.fb_in1k deit3_large_patch16_224.fb_in22k_ft_in1k"
# SWIN="swin_base_patch4_window7_224.ms_in1k swinv2_base_window12_192.ms_in22k swin_s3_base_224.ms_in1k"
SWIN="swinv2_base_window12_192.ms_in22k swin_s3_base_224.ms_in1k"
FOCALNET="focalnet_base_lrf focalnet_large_fl3.ms_in22k focalnet_xlarge_fl3.ms_in22k"


# decomposition (familiar)
uv run python -m scripts.decomposition \
  --annotations "data/datasets/full/low_mid_level_vision/decomposition/annotation.csv" \
  --shape_type "familiar" \
  --models $RESNET50 $RESNEXT101 $CONVNEXT $VIT $DEIT $SWIN $FOCALNET  --overwrite_recordings

# # decomposition (unfamiliar)
# uv run python -m scripts.decomposition \
#   --annotations "data/datasets/full/low_mid_level_vision/decomposition/annotation.csv" \
#   --shape_type "unfamiliar" \
#   --models $RESNET50 $RESNEXT101 $CONVNEXT $VIT $DEIT $SWIN $FOCALNET
