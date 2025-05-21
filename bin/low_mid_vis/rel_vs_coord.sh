#!/bin/bash

# RESNET="resnet50s.gluon_in1k resnet101.gluon_in1"
# RESNEXT101="resnext101_32x4d.fb_swsl_ig1b_ft_in1k resnext101_32x8d.fb_swsl_ig1b_ft_in1k"
# CONVNEXT="convnext_base.clip_laion2b_augreg_ft_in1k convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384 convnext_xlarge.fb_in22k_ft_in1k"
# VIT="vit_base_patch16_clip_224.openai_ft_in12k_in1k vit_large_patch14_clip_224.laion2b_ft_in12k_in1k vit_large_patch14_clip_224.openai_ft_in12k_in1k"
# DEIT="deit3_base_patch16_224.fb_in1k deit3_medium_patch16_224.fb_in1k deit3_large_patch16_224.fb_in22k_ft_in1k"
# FOCALNET="focalnet_base_lrf.ms_in1k focalnet_base_srf.ms_in1k"


# ImageNet-only models
CONVNEXT="convnext_tiny_imagenet_full_seed-0 convnext_base_imagenet_full_seed-0 convnext_large_imagenet_full_seed-0"
DEIT="deit_base_imagenet_full_seed-0 deit_large_imagenet_full_seed-0"

uv run python -m scripts.low_mid_vis.rel_vs_coord \
  --annotations "data/datasets/full/low_mid_level_vision/relational_vs_coordinate/annotation.csv" \
  --models $CONVNEXT $DEIT --overwrite_recordings
  # --models $RESNET $RESNEXT $CONVNEXT $VIT $DEIT $SWIN $FOCALNET
