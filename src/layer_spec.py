import logging
import re
from typing import Literal, Sequence
import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)

VALID_STREAMS = ("in", "out", "delta", "all")
StreamType = Literal["in", "out", "delta"]


def parse_layer_spec(spec: str) -> tuple[str, list[StreamType], bool]:
    """Parse a layer target specification into (pattern, list_of_streams, explicit_stream).

    Supported stream suffixes:
      - ':in'    -> record input[0]
      - ':out'   -> record output
      - ':delta' -> record output - input[0]
      - ':all'   -> record all three (in, out, delta)

    If no valid suffix is present, defaults to ['out'] with explicit_stream=False.
    """
    spec = spec.strip()
    if ":" in spec:
        base, suffix = spec.rsplit(":", 1)
        suffix = suffix.strip()
        if suffix in VALID_STREAMS:
            if suffix == "all":
                return base.strip(), ["in", "out", "delta"], True
            return base.strip(), [suffix], True  # type: ignore
    return spec, ["out"], False


def sanitize_key_for_module_dict(key: str) -> str:
    """Sanitize a layer key for PyTorch nn.ModuleDict, which forbids '.' in keys."""
    return key.replace(".", "__")


def get_leaf_modules_with_legacy_names(model: nn.Module) -> list[tuple[str, str, nn.Module]]:
    """Return list of (legacy_name, module_path, module) for all leaf modules.

    legacy_name is formatted as f"{idx}: {type(layer).__name__}".
    """
    leaves = []
    idx = 0

    def collect(mod: nn.Module, prefix: str = ""):
        nonlocal idx
        children = list(mod.named_children())
        if children:
            for name, child in children:
                sub_prefix = f"{prefix}.{name}" if prefix else name
                collect(child, sub_prefix)
        else:
            legacy_name = f"{idx}: {type(mod).__name__}"
            leaves.append((legacy_name, prefix, mod))
            idx += 1

    collect(model)
    return leaves


def resolve_layer_targets(
    model: nn.Module,
    patterns: Sequence[str],
) -> list[tuple[str, nn.Module, StreamType]]:
    r"""Resolve layer patterns into a list of (target_key, module, stream_type).

    Supports:
      - Exact module paths (e.g. 'stages.0.blocks.0', 'stem')
      - Regex module paths (e.g. r'stages\.\d+\.blocks\.\d+')
      - Stream suffixes (e.g. ':in', ':out', ':delta', ':all')
      - Legacy leaf names (e.g. '^0: Conv2d$', 'Conv2d', 'Linear')
    """
    named_mods = dict(model.named_modules())
    named_mods.pop("", None)  # exclude root module

    leaf_records = get_leaf_modules_with_legacy_names(model)

    resolved: list[tuple[str, nn.Module, StreamType]] = []
    seen_keys: set[str] = set()

    for spec in patterns:
        base_pattern, streams, explicit_stream = parse_layer_spec(spec)
        matched = False

        # 1. Exact match against module path
        if base_pattern in named_mods:
            mod = named_mods[base_pattern]
            matched = True
            for s in streams:
                key = f"{base_pattern}:{s}" if explicit_stream else base_pattern
                if key not in seen_keys:
                    seen_keys.add(key)
                    resolved.append((key, mod, s))
            continue

        # 2. Regex match against module paths (prefer fullmatch)
        try:
            regex = re.compile(base_pattern)
        except re.error as e:
            _logger.warning(f"Invalid regex pattern '{base_pattern}': {e}")
            continue

        # Check fullmatch on module paths first
        matched_mods = [(path, mod) for path, mod in named_mods.items() if regex.fullmatch(path)]
        if not matched_mods:
            # Check search on module paths
            matched_mods = [(path, mod) for path, mod in named_mods.items() if regex.search(path)]

        if matched_mods:
            matched = True
            for mod_path, mod in matched_mods:
                for s in streams:
                    key = f"{mod_path}:{s}" if explicit_stream else mod_path
                    if key not in seen_keys:
                        seen_keys.add(key)
                        resolved.append((key, mod, s))

        # 3. Fallback: match against legacy leaf names (e.g. '0: Conv2d' or 'Conv2d')
        if not matched:
            for legacy_name, mod_path, mod in leaf_records:
                if regex.search(legacy_name) or regex.fullmatch(legacy_name):
                    matched = True
                    for s in streams:
                        key = f"{legacy_name}:{s}" if explicit_stream else legacy_name
                        if key not in seen_keys:
                            seen_keys.add(key)
                            resolved.append((key, mod, s))

        if not matched:
            _logger.warning(f"Pattern '{spec}' (base: '{base_pattern}') matched no modules in model.")

    return resolved


def extract_stream_tensor(
    inp: tuple | list | torch.Tensor,
    out: tuple | list | torch.Tensor,
    stream: StreamType,
    layer_name: str = "",
) -> torch.Tensor:
    """Extract the specified stream tensor (:in, :out, :delta) from hook args."""
    inp_tensor = inp[0] if isinstance(inp, (tuple, list)) and len(inp) > 0 else inp
    out_tensor = out[0] if isinstance(out, (tuple, list)) and len(out) > 0 else out

    if not isinstance(inp_tensor, torch.Tensor) or not isinstance(out_tensor, torch.Tensor):
        if stream == "in" and isinstance(inp_tensor, torch.Tensor):
            return inp_tensor
        if stream == "out" and isinstance(out_tensor, torch.Tensor):
            return out_tensor
        raise TypeError(
            f"Hook on '{layer_name}' received non-Tensor input/output: "
            f"inp={type(inp_tensor)}, out={type(out_tensor)}"
        )

    if stream == "in":
        return inp_tensor
    elif stream == "out":
        return out_tensor
    elif stream == "delta":
        if inp_tensor.shape != out_tensor.shape:
            raise ValueError(
                f"Cannot compute ':delta' for '{layer_name}': input shape {inp_tensor.shape} "
                f"does not match output shape {out_tensor.shape}. ':delta' requires identical shapes."
            )
        return out_tensor - inp_tensor
    else:
        raise ValueError(f"Unknown stream type: {stream}")
