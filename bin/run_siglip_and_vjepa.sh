#!/bin/bash
# ==============================================================================
# Benchmarking SigLIP and V-JEPA across all Low- and Mid-Level Vision Tasks
#
# Models evaluated:
#   1. vit_so400m_patch14_siglip_378.v2_webli  (SigLIP2 ViT-So400M)
#   2. vjepa2_1_vit_large_384                  (V-JEPA 2.1 ViT-Large)
#
# Experiments evaluated:
#   1. amodal_completion
#   2. crowding (shapes and distributions)
#   3. decomposition
#   4. depth_drawings
#   5. emergent_features (linearity, orientation, proximity)
#   6. nap_vs_mp
#   7. rel_vs_coord
#
# For each experiment, both 'pre_act' and 'res_stream' streams are recorded.
# ==============================================================================

set -uo pipefail

RESULTS_DIR="${1:-data/results}"

MODELS=(
  "vit_so400m_patch14_siglip_378.v2_webli"
  "vjepa2_1_vit_large_384"
)

# Helper: locate dataset annotation file among candidate paths
resolve_annotations() {
  local candidate
  for candidate in "$@"; do
    if [ -f "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done
  echo "$1"
  return 1
}

# Helper: retrieve model-specific layer patterns for pre_act and res_stream
get_model_patterns() {
  local model="$1"
  case "$model" in
    vit_so400m_patch14_siglip_378.v2_webli)
      PRE_ACT='^blocks\.([0-9]|[1-2][0-9])\.norm[12]:in ^norm:in ^attn_pool(?:\.proj_drop|)'
      RES_STREAM='^blocks\.([0-9]|[1-2][0-9])\.drop_path2'
      ;;
    vjepa2_1_vit_large_384)
      PRE_ACT='^blocks\.([0-9]|[1-2][0-9])'
      RES_STREAM='^blocks\.([0-9]|[1-2][0-9])\.drop_path'
      ;;
    *)
      PRE_ACT='^blocks\.([0-9]|[1-9][0-9]):out'
      RES_STREAM='^blocks\.([0-9]|[1-9][0-9])\.drop_path2'
      ;;
  esac
}

# Helper: run a single recording command
run_recording() {
  local task="$1" model="$2" annot_file="$3" results="$4" record="$5" tag="$6"

  if [ ! -f "$annot_file" ]; then
    echo "⚠️  [Skip] Annotation file not found: $annot_file"
    return 0
  fi

  echo "▶ Running <$task> on <$model> (tag: $tag)"
  echo "  Annotations : $annot_file"
  echo "  Results dir : $results"
  echo "  Patterns    : $record"

  uv run python -m "scripts.low_mid_vis.$task" \
    --annotations_file "$annot_file" \
    --models "$model" \
    --record_from $record \
    --overwrite_recordings \
    --results_folder "$results" \
    --output_tag "$tag"
}

# Helper: run both pre_act and res_stream for a task
run_both_streams() {
  local task="$1" model="$2" annot_file="$3" results="$4" pre_act="$5" res_stream="$6"
  run_recording "$task" "$model" "$annot_file" "$results" "$pre_act" "pre_act"
  run_recording "$task" "$model" "$annot_file" "$results" "$res_stream" "res_stream"
}

# Run all 7 experiments for a single model
run_all_experiments() {
  local model="$1"
  local results="$2"

  local PRE_ACT RES_STREAM
  get_model_patterns "$model"

  echo ""
  echo "===================================================================="
  echo "🚀 Benchmarking Model: $model"
  echo "   pre_act    : $PRE_ACT"
  echo "   res_stream : $RES_STREAM"
  echo "===================================================================="

  # 1. Amodal Completion
  local amodal_annot
  amodal_annot=$(resolve_annotations "data/datasets/low_mid_vision/amodal_completion/annotation.csv")
  run_both_streams "amodal_completion" "$model" "$amodal_annot" "$results" "$PRE_ACT" "$RES_STREAM"

  # 2. Visual Crowding (Shapes & Distributions)
  local crowd_shapes_annot crowd_dist_annot
  crowd_shapes_annot=$(resolve_annotations \
    "data/datasets/low_mid_vision/uncrowding_shapes_large_offset/annotation.csv" \
    "data/datasets/low_mid_vision/uncrowding_shapes/annotation.csv")
  run_both_streams "crowding" "$model" "$crowd_shapes_annot" "$results/uncrowding_shapes" "$PRE_ACT" "$RES_STREAM"

  crowd_dist_annot=$(resolve_annotations \
    "data/datasets/low_mid_vision/uncrowding_distributions_large_offset/annotation.csv" \
    "data/datasets/low_mid_vision/uncrowding_distributions/annotation.csv")
  run_both_streams "crowding" "$model" "$crowd_dist_annot" "$results/uncrowding_distributions" "$PRE_ACT" "$RES_STREAM"

  # 3. Shape Decomposition
  local decomp_annot
  decomp_annot=$(resolve_annotations "data/datasets/low_mid_vision/decomposition/annotation.csv")
  run_both_streams "decomposition" "$model" "$decomp_annot" "$results" "$PRE_ACT" "$RES_STREAM"

  # 4. Depth Drawings
  local depth_annot
  depth_annot=$(resolve_annotations "data/datasets/low_mid_vision/depth_drawings/annotation.csv")
  run_both_streams "depth_drawings" "$model" "$depth_annot" "$results" "$PRE_ACT" "$RES_STREAM"

  # 5. Emergent Gestalt Features (Linearity, Orientation, Proximity)
  local ef_lin_annot ef_ori_annot ef_prox_annot
  ef_lin_annot=$(resolve_annotations "data/datasets/low_mid_vision/emergent_features/linearity/annotation.csv")
  run_both_streams "emergent_features" "$model" "$ef_lin_annot" "$results" "$PRE_ACT" "$RES_STREAM"

  ef_ori_annot=$(resolve_annotations "data/datasets/low_mid_vision/emergent_features/orientation/annotation.csv")
  run_both_streams "emergent_features" "$model" "$ef_ori_annot" "$results" "$PRE_ACT" "$RES_STREAM"

  ef_prox_annot=$(resolve_annotations "data/datasets/low_mid_vision/emergent_features/proximity/annotation.csv")
  run_both_streams "emergent_features" "$model" "$ef_prox_annot" "$results" "$PRE_ACT" "$RES_STREAM"

  # 6. Non-Accidental Properties (NAP) vs. Metric Properties (MP)
  local nap_annot
  nap_annot=$(resolve_annotations "data/datasets/low_mid_vision/nap_vs_mp_3d_geons/annotation.csv")
  run_both_streams "nap_vs_mp" "$model" "$nap_annot" "$results" "$PRE_ACT" "$RES_STREAM"

  # 7. Relational vs. Coordinate Representation
  local rel_annot
  rel_annot=$(resolve_annotations \
    "data/datasets/low_mid_vision/relational_vs_coordinate/annotation.csv" \
    "data/datasets/low_mid_vision/rel_vs_coord_change/annotation.csv" \
    "data/low_mid_vision/rel_vs_coord_change/annotation.csv")
  run_both_streams "rel_vs_coord" "$model" "$rel_annot" "$results" "$PRE_ACT" "$RES_STREAM"
}

# Main Execution Loop
for model in "${MODELS[@]}"; do
  run_all_experiments "$model" "$RESULTS_DIR"
done

echo ""
echo "✅ All experiments for SigLIP and V-JEPA finished. Results saved in <$RESULTS_DIR>."
