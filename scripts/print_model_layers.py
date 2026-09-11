import argparse
import sys
from pathlib import Path
from typing import Optional
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import init_model, setup_logging
from src.layer_spec import resolve_layer_targets, get_leaf_modules_with_legacy_names

_logger = setup_logging(__name__)


def get_leaf_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Extract all leaf modules (modules without children) in depth-first order.

    Maintained for backward compatibility. Returns a list of (module_path, module).
    """
    records = get_leaf_modules_with_legacy_names(model)
    return [(path, mod) for _, path, mod in records]


def print_hierarchy(model: nn.Module, max_depth: int):
    """Print model hierarchy up to max_depth levels."""
    named_mods = dict(model.named_modules())
    named_mods.pop("", None)

    print(f"{'Depth':<6} {'Module Path':<50} {'Class':<25} {'Params':>12}")
    print("-" * 97)

    count = 0
    for path, mod in named_mods.items():
        depth = path.count(".") + 1
        if depth <= max_depth:
            count += 1
            param_count = sum(p.numel() for p in mod.parameters(recurse=False))
            param_str = f"{param_count:,}" if param_count > 0 else "-"
            indent = "  " * (depth - 1)
            display_path = f"{indent}{path}"
            print(f"{depth:<6} {display_path:<50} {type(mod).__name__:<25} {param_str:>12}")

    print("-" * 97)
    print(f"Displayed {count} modules up to depth {max_depth}.\n")


def print_pattern_matches(model: nn.Module, patterns: list[str]):
    """Print all targets resolved by the provided layer patterns/regexes."""
    resolved = resolve_layer_targets(model, patterns)
    print(f"Filter patterns: {', '.join(patterns)}")
    print(f"Resolved targets ({len(resolved)} total):")
    print(f"{'Target Key':<50} {'Class':<25} {'Stream':<8}")
    print("-" * 85)
    for key, mod, stream in resolved:
        print(f"{key:<50} {type(mod).__name__:<25} {stream:<8}")
    print("-" * 85)
    print(f"Total resolved targets: {len(resolved)}\n")


def print_leaf_layers(model: nn.Module, pattern: Optional[str] = None, show_path: bool = False):
    """Print leaf layers in legacy index: Type format."""
    leaf_records = get_leaf_modules_with_legacy_names(model)
    total_layers = len(leaf_records)

    print(f"Total leaf layers: {total_layers}")
    if pattern:
        print(f"Filter pattern: {pattern}")
    print("-" * 60)

    import re
    match_count = 0
    for legacy_name, path, layer in leaf_records:
        if pattern is None or re.search(pattern, legacy_name):
            match_count += 1
            if show_path:
                print(f"{legacy_name:<35} # {path}")
            else:
                print(legacy_name)

    print("-" * 60)
    if pattern:
        print(f"Matched {match_count} of {total_layers} layers.\n")
    else:
        print(f"Printed {total_layers} layers.\n")


def print_model_layers(
    model_name: str,
    patterns: Optional[list[str]] = None,
    depth: Optional[int] = None,
    leaf: bool = False,
    show_path: bool = False,
):
    """Load model and inspect its architecture."""
    _logger.info(f"Loading model: <{model_name}>...")
    model = init_model(model_name)

    print(f"\n{'='*80}\nModel: {model_name}\n{'='*80}")

    if patterns:
        print_pattern_matches(model, patterns)
    elif depth is not None:
        print_hierarchy(model, max_depth=depth)
    elif leaf:
        print_leaf_layers(model, pattern=None, show_path=show_path)
    else:
        # Default: show depth 3 hierarchy to easily locate blocks
        print("Tip: Use --depth <N> to view hierarchy, or --pattern to test regex targeting.\n")
        print_hierarchy(model, max_depth=3)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect model architecture layers, test regex patterns, and view hierarchy."
    )
    parser.add_argument(
        "--model",
        "--models",
        type=str,
        nargs="+",
        dest="models",
        required=True,
        help="One or more model architecture names (e.g. convnext_tiny, resnet50)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Max hierarchy depth to display (e.g. 2 or 3 to inspect macro blocks)",
    )
    parser.add_argument(
        "--pattern",
        "--patterns",
        type=str,
        nargs="+",
        dest="patterns",
        default=None,
        help="Regex pattern(s) to test target resolution (e.g. 'stages\\.\\d+\\.blocks\\.\\d+:all')",
    )
    parser.add_argument(
        "--leaf",
        action="store_true",
        help="Print flat leaf layers in legacy '0: Conv2d' format",
    )
    parser.add_argument(
        "--show_path",
        action="store_true",
        help="Show module path alongside legacy leaf format",
    )
    args = parser.parse_args()

    for model_name in args.models:
        print_model_layers(
            model_name=model_name,
            patterns=args.patterns,
            depth=args.depth,
            leaf=args.leaf,
            show_path=args.show_path,
        )


if __name__ == "__main__":
    main()
