#!/bin/bash

MODELS=(
  # # ResNet
  resnet50s.gluon_in1k
  resnet101.gluon_in1k
)

uv run python -m scripts.low_mid_vis.crowding \
  --annotations_file "data/datasets/low_mid_vision/un_crowding/annotation.csv" \
  --models "${MODELS[@]}" \
  --record_from "conv1.(2|5)" "act1"  "layer[1-4]\.[1-22]\.act3:out" \
  --overwrite_recordings \
  --results_folder data/results/un_crowding/layers-0-3-13
