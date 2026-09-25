#!/bin/bash

# NOTES:
# For ResNet50s/101, each top-level layer corresponds to a downsampling step. Within each of
# these steps we then perform several residual + aggregation computations, where the first such
# operation within each layer also expands the number of channels. This is used to compensate
# for the increase information that each patch is encoding. In general


MODELS=(
  # "resnet50s.gluon_in1k"
  # "resnet101.gluon_in1k"
  "resnext101_32x8d.fb_swsl_ig1b_ft_in1k"
  "resnext101_32x4d.fb_swsl_ig1b_ft_in1k"
)

RECORD_FROM=(
  # "^act1:out ^layer[1-4]\.[0-5]\.act3:out ^fc:out"
  # "^act1:out ^layer[1-4]\.[0-9]\.act3:out ^layer[1-4]\.[0-4][0-9]\.act3:out ^fc:out"
  "^act1:out ^layer[1-4]\.[0-9]\.act3:out ^layer[1-4]\.[0-4][0-9]\.act3:out ^fc:out"
  "^act1:out ^layer[1-4]\.[0-9]\.act3:out ^layer[1-4]\.[0-4][0-9]\.act3:out ^fc:out"
)

for i in "${!MODELS[@]}"; do
uv run python -m scripts.low_mid_vis.amodal_completion \
  --annotations_file "data/datasets/low_mid_vision/amodal_completion/annotation.csv" \
  --models ${MODELS[$i]} \
  --record_from ${RECORD_FROM[$i]} \
  --overwrite_recordings \
  --results_folder data/results \
  --output_tag 'post_act'
done


RECORD_FROM=(
  # "^act1:in ^layer[1-4]\.[0-5]\.act3:in ^fc:in"
  # "^act1:in ^layer[1-4]\.[0-9]\.act3:in ^layer[1-4]\.[0-4][0-9]\.act3:in ^fc:in"
  "^act1:in ^layer[1-4]\.[0-9]\.act3:in ^layer[1-4]\.[0-4][0-9]\.act3:in ^fc:in"
  "^act1:in ^layer[1-4]\.[0-9]\.act3:in ^layer[1-4]\.[0-4][0-9]\.act3:in ^fc:in"
)

for i in "${!MODELS[@]}"; do
uv run python -m scripts.low_mid_vis.amodal_completion \
  --annotations_file "data/datasets/low_mid_vision/amodal_completion/annotation.csv" \
  --models ${MODELS[$i]} \
  --record_from ${RECORD_FROM[$i]} \
  --overwrite_recordings \
  --results_folder data/results \
  --output_tag 'pre_act'
done


RECORD_FROM=(
  # "^layer[1-4]\.[0-5]\.bn3:out"
  # "^layer[1-4]\.[0-9]\.bn3:out ^layer[1-4]\.[0-4][0-9]\.bn3:out"
  "^layer[1-4]\.[0-9]\.bn3:out ^layer[1-4]\.[0-4][0-9]\.bn3:out"
  "^layer[1-4]\.[0-9]\.bn3:out ^layer[1-4]\.[0-4][0-9]\.bn3:out"
)

for i in "${!MODELS[@]}"; do
uv run python -m scripts.low_mid_vis.amodal_completion \
  --annotations_file "data/datasets/low_mid_vision/amodal_completion/annotation.csv" \
  --models ${MODELS[i]} \
  --record_from  ${RECORD_FROM[$i]} \
  --overwrite_recordings \
  --results_folder data/results \
  --output_tag 'res_stream'
done
