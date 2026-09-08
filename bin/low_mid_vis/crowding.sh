#!/bin/bash

MODELS=(
  # # ResNet
  # resnet50s.gluon_in1k
  # resnet101.gluon_in1k

  # # ResNeXt
  # resnext101_32x4d.fb_swsl_ig1b_ft_in1k
  # resnext101_32x8d.fb_swsl_ig1b_ft_in1k

  # ConvNeXt: See the chunked version as it was impossible to do this in one go.

  # ViT: See the chunked version as it was impossible to do this in one go.

  # Swin: See the chunked version as it was impossible to do this in one go.

  # # ImageNet-only models
  # convnext_tiny.fb_in1k
  # convnext_base.fb_in1k
  # convnext_large.fb_in1k
  # deit3_base_patch16_224.fb_in1k
  # deit3_large_patch16_224.fb_in1k
)

uv run python -m scripts.low_mid_vis.crowding \
  --annotations_file "data/datasets/low_mid_vision/un_crowding/annotation.csv" \
  --models "${MODELS[@]}" \
  --record_from "Conv2d" "Linear" \
  --overwrite_recordings \
  --results_folder data/results/un_crowding/layers-0-3-13
