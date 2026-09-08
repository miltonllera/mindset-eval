import argparse
import math
import os
import re
from pathlib import Path
from typing import Optional

from src.utils import init_model, setup_logging
from scripts.print_model_layers import get_leaf_layers

_logger = setup_logging(__name__)


def partition_layers(
    layers: list[str],
    num_chunks: Optional[int] = None,
    chunk_size: Optional[int] = None,
) -> list[list[str]]:
    """Split a list of layer names into chunks."""
    total = len(layers)
    if total == 0:
        return []

    if chunk_size is not None and chunk_size > 0:
        return [layers[i : i + chunk_size] for i in range(0, total, chunk_size)]

    if num_chunks is not None and num_chunks > 0:
        num_chunks = min(num_chunks, total)
        k = math.ceil(total / num_chunks)
        chunks = []
        for i in range(num_chunks):
            start = i * k
            end = min(start + k, total)
            if start < total:
                chunks.append(layers[start:end])
        return chunks

    raise ValueError("Either num_chunks or chunk_size must be specified.")


def generate_bash_script(
    model_name: str,
    experiment_script: str,
    patterns: list[str],
    num_chunks: Optional[int] = None,
    chunk_size: Optional[int] = None,
    annotations_file: str = "data/datasets/low_mid_vision/un_crowding/annotation.csv",
    results_folder: Optional[str] = None,
) -> str:
    """Extract layers matching patterns and generate a chunked bash script."""
    _logger.info(f"Loading model <{model_name}> to inspect leaf layers...")
    model = init_model(model_name)
    leaf_layers = get_leaf_layers(model)

    # Format names as FeatureDecoder does: f"{idx}: {type(layer).__name__}"
    formatted_layers = [f"{idx}: {type(layer).__name__}" for idx, (_, layer) in enumerate(leaf_layers)]

    # Filter layers by the provided patterns
    matched_layers = []
    for name in formatted_layers:
        if any(re.search(p, name) for p in patterns):
            matched_layers.append(name)

    _logger.info(
        f"Found {len(matched_layers)} layers matching patterns {patterns} out of {len(formatted_layers)} leaf layers."
    )

    chunks = partition_layers(matched_layers, num_chunks=num_chunks, chunk_size=chunk_size)
    total_chunks = len(chunks)

    # Normalize experiment script (e.g. convert 'scripts/low_mid_vis/crowding.py' -> 'scripts.low_mid_vis.crowding')
    module_name = experiment_script.replace("/", ".").replace("\\", ".")
    if module_name.endswith(".py"):
        module_name = module_name[:-3]

    lines = [
        "#!/bin/bash",
        "# ==============================================================================",
        f"# Automatically generated chunked execution script",
        f"# Target model:        {model_name}",
        f"# Experiment script:   {module_name}",
        f"# Filter patterns:     {', '.join(patterns)}",
        f"# Total matched layers: {len(matched_layers)}",
        f"# Total chunks:        {total_chunks}",
        "# ==============================================================================",
        "set -euo pipefail",
        "",
        "MODELS=(",
        f"  {model_name}",
        ")",
        "",
        f'ANNOTATIONS_FILE="{annotations_file}"',
    ]

    if results_folder:
        lines.append(f'RESULTS_FOLDER="{results_folder}"')
    lines.append("")

    for chunk_idx, chunk in enumerate(chunks):
        tag = f"chunk_{chunk_idx}"
        first_layer = chunk[0]
        last_layer = chunk[-1]

        lines.append("# " + "-" * 78)
        lines.append(f"# Chunk {chunk_idx + 1}/{total_chunks}: {len(chunk)} layers ({first_layer} -> {last_layer})")
        lines.append("# " + "-" * 78)
        lines.append(f"echo \"[$(date +'%Y-%m-%d %H:%M:%S')] Starting Chunk {chunk_idx + 1}/{total_chunks} ({tag})...\"")
        lines.append(f"uv run python -m {module_name} \\")
        lines.append('  --annotations_file "${ANNOTATIONS_FILE}" \\')
        lines.append('  --models "${MODELS[@]}" \\')
        if results_folder:
            lines.append('  --results_folder "${RESULTS_FOLDER}" \\')
        lines.append(f'  --output_tag "{tag}" \\')
        lines.append("  --overwrite_recordings \\")
        lines.append("  --record_from \\")
        for layer_name in chunk:
            lines.append(f'    "^{layer_name}$" \\')
        # Remove trailing slash from the last argument in the block
        lines[-1] = lines[-1][:-2]
        lines.append("")

    lines.append('echo "[$(date +\'%Y-%m-%d %H:%M:%S\')] All chunks completed successfully!"')
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a chunked bash script for layer-partitioned experiment runs."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to inspect (e.g. convnext_base.clip_laion2b_augreg_ft_in1k)",
    )
    parser.add_argument(
        "--script",
        type=str,
        default="scripts.low_mid_vis.crowding",
        help="Experiment module or script path to invoke (default: scripts.low_mid_vis.crowding)",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        nargs="+",
        default=["Conv2d", "Linear"],
        help="Layer types/regex patterns to extract (default: Conv2d Linear)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--num_chunks",
        type=int,
        help="Number of chunks to partition layers into",
    )
    group.add_argument(
        "--chunk_size",
        type=int,
        help="Number of layers per chunk",
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        default="data/datasets/low_mid_vision/un_crowding/annotation.csv",
        help="Path to the annotations file",
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default=None,
        help="Optional custom results folder to pass to the script",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save the generated bash script (if not provided, prints to stdout)",
    )

    args = parser.parse_args()

    bash_content = generate_bash_script(
        model_name=args.model,
        experiment_script=args.script,
        patterns=args.patterns,
        num_chunks=args.num_chunks,
        chunk_size=args.chunk_size,
        annotations_file=args.annotations_file,
        results_folder=args.results_folder,
    )

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(bash_content)
        os.chmod(output_path, 0o755)
        _logger.info(f"Generated executable bash script saved to: <{output_path}>")
    else:
        print(bash_content)


if __name__ == "__main__":
    main()
