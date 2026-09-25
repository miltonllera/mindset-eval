#!/bin/bash

# NOTES:
# For ResNet50s/101, each top-level layer corresponds to a downsampling step. Within each of
# these steps we then perform several residual + aggregation computations, where the first such
# operation within each layer also expands the number of channels. This is used to compensate
# for the increase information that each patch is encoding. In general


MODELS=(
  # "resnet50s.gluon_in1k"
  # "resnet101.gluon_in1k" "resnext101_32x8d.fb_swsl_ig1b_ft_in1k" "resnext101_32x4d.fb_swsl_ig1b_ft_in1k"
  # "convnext_tiny.fb_in1k"
  # "convnext_base.clip_laion2b_augreg_ft_in1k convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384 convnext_xlarge.fb_in22k_ft_in1k"
  # "vit_base_patch16_clip_224.openai_ft_in12k_in1k vit_large_patch14_clip_224.laion2b_ft_in12k_in1k vit_large_patch14_clip_224.openai_ft_in12k_in1k"
  # "deit3_base_patch16_224.fb_in1k deit3_medium_patch16_224.fb_in1k deit3_large_patch16_224.fb_in22k_ft_in1k"
  # "focalnet_base_lrf.ms_in1k focalnet_base_srf.ms_in1k"
  "swin_base_patch4_window7_224.ms_in1k swin_s3_base_224.ms_in1k swinv2_base_window12to16_192to256.ms_in22k_ft_in1k"
)

# RECORD_FROM=(
#   "^act1:out ^layer[1-4]\.[0-5]\.act3:out ^fc:out"
#   "^act1:out ^layer[1-4]\.[0-9]\.act3:out ^layer[1-4]\.[1-2][0-9]\.act3:out ^fc:out"
#   "^stem\.1:out ^stages\.[0-3]\.blocks\.[0-9]:out"
#   "^stem\.1:out ^stages\.[0-3]\.blocks\.[0-9]:out  ^stages\.[0-3]\.blocks\.[1-2][0-9]:out"
#   ""
# )

# for i in "${!MODELS[@]}"; do
# uv run python -m scripts.low_mid_vis.amodal_completion \
#   --annotations_file "data/datasets/low_mid_vision/amodal_completion/annotation.csv" \
#   --models ${MODELS[$i]} \
#   --record_from ${RECORD_FROM[$i]} \
#   --overwrite_recordings \
#   --results_folder data/results \
#   --output_tag 'post_act'
# done


RECORD_FROM=(
  # "^act1:in ^layer[1-4]\.[0-5]\.act3:in ^fc:in"
  # "^act1:in ^layer[1-4]\.[0-9]\.act3:in ^layer[1-4]\.[1-2][0-9]\.act3:in ^fc:in"
  # "^stem\.1:out ^stages\.[0-3]\.blocks\.[0-9]:out"
  # "^stem\.1:out ^stages\.[0-3]\.blocks\.[0-9]:out  ^stages\.[0-3]\.blocks\.[1-2][0-9]:out"
  # "^blocks\.[0-9]:out ^blocks\.[1-9][0-9]:out"
  # "^layers\.[0-3]\.blocks\.([0-9]|1[0-9])"
  "^layers\.[0-3]\.blocks\.([0-9]|[1-3][0-9])"
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
  # "^layer[1-4]\.[0-9]\.bn3:out ^layer[1-4]\.[1-2][0-9]\.bn3:out"
  # "^stages\.[0-3]\.blocks\.[0-9].drop_path:out"
  # "^stages\.[0-3]\.blocks\.[0-9].drop_path:out  ^stages\.[0-3]\.blocks\.[1-2][0-9].drop_path:out"
  # "^blocks\.[0-9]\.drop_path2 ^blocks\.[1-9][0-9]\.drop_path2"
  # "^layers\.[0-3]\.blocks\.([0-9]|1[0-9]):in"
  "^layers\.[0-3]\.blocks\.([0-9]|[1-3][0-9]):in"
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
