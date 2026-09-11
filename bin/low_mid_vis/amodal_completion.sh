#!/bin/bash

MODELS=(
  # # ResNet
  resnet50s.gluon_in1k
  resnet101.gluon_in1k
)

# uv run python -m scripts.low_mid_vis.amodal_completion \
#   --annotations_file "data/datasets/low_mid_vision/amodal_completion/annotation.csv" \
#   --models "${MODELS[@]}" \
#   --record_from "^act1" "^layer[1-4]\.[0-5]\.act3:out" \
#                 "^layer[1-4]\.[0-4][0-9]\.act3:out" \
#   --overwrite_recordings \
#   --results_folder data/results/post_act/


# uv run python -m scripts.low_mid_vis.amodal_completion \
#   --annotations_file "data/datasets/low_mid_vision/amodal_completion/annotation.csv" \
#   --models "${MODELS[@]}" \
#   --record_from "^conv(1|\.6)" "^layer[1-4]\.[0-5]\.act3:in" \
#                 "^layer[1-4]\.[0-4][0-9]\.act3:in" \
#   --overwrite_recordings \
#   --results_folder data/results/pre_act/


uv run python -m scripts.low_mid_vis.amodal_completion \
  --annotations_file "data/datasets/low_mid_vision/amodal_completion/annotation.csv" \
  --models "${MODELS[@]}" \
  --record_from  "^layer[1-4]\.[0-5]\.bn3:out" "^layer[1-4]\.[0-4][0-9]\.bn3:out" \
  --overwrite_recordings \
  --results_folder data/results/main_stream/
