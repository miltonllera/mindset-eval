import argparse
import re
from typing import Optional
import torch.nn as nn

from src.utils import init_model, setup_logging

_logger = setup_logging(__name__)


def get_leaf_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Extract all leaf modules (modules without children) in depth-first order.

    Returns a list of tuples containing (module_path, module).
    """
    layers = []

    def collect(module: nn.Module, prefix: str = ""):
        children = list(module.named_children())
        if children:
            for name, child in children:
                sub_prefix = f"{prefix}.{name}" if prefix else name
                collect(child, sub_prefix)
        else:
            layers.append((prefix, module))

    collect(model)
    return layers


def print_model_layers(
    model_name: str,
    pattern: Optional[str] = None,
    show_path: bool = False,
):
    """Load a model and print its leaf layers formatted as in FeatureDecoder._register_hooks."""
    _logger.info(f"Loading model: <{model_name}>...")
    model = init_model(model_name)

    leaf_layers = get_leaf_layers(model)
    total_layers = len(leaf_layers)

    print(f"\nModel: {model_name} (Total leaf layers: {total_layers})")
    if pattern:
        print(f"Filter pattern: {pattern}")
    print("-" * 60)

    match_count = 0
    for idx, (path, layer) in enumerate(leaf_layers):
        name = f"{idx}: {type(layer).__name__}"
        if pattern is None or re.search(pattern, name):
            match_count += 1
            if show_path:
                print(f"{name:<35} # {path}")
            else:
                print(name)

    print("-" * 60)
    if pattern:
        print(f"Matched {match_count} of {total_layers} layers.\n")
    else:
        print(f"Printed {total_layers} layers.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and print model leaf layers in the format expected by FeatureDecoder."
    )
    parser.add_argument(
        "--model",
        "--models",
        type=str,
        nargs="+",
        dest="models",
        required=True,
        help="One or more model architecture names (e.g. resnet50s.gluon_in1k)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Optional regex pattern to filter layer names (matching FeatureDecoder.decode_from)",
    )
    parser.add_argument(
        "--show_path",
        action="store_true",
        help="Also show module path (e.g. layer1.0.conv1) alongside the index: Type format",
    )
    args = parser.parse_args()

    for model_name in args.models:
        print_model_layers(
            model_name=model_name,
            pattern=args.pattern,
            show_path=args.show_path,
        )


if __name__ == "__main__":
    main()
