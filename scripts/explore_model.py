"""Interactive Model Architecture Explorer.

A lightweight, zero-extra-dependency local web app and CLI tool to interactively explore,
expand/collapse, inspect connections, and view tensor input/output shapes for PyTorch / timm models,
with full functional operation tracing (add, matmul, attention, cat) and extensibility for VLMs and JEPA architectures.

Usage:
    uv run python scripts/explore_model.py
    uv run python scripts/explore_model.py --model vit_base_patch16_clip_224.openai_ft_in12k_in1k
    uv run python scripts/explore_model.py --port 8080 --no-browser
"""

import argparse
import http.server
import importlib.util
import json
import os
import re
import socket
import sys
import urllib.parse
import webbrowser
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None

try:
    from torchview import draw_graph
except ImportError:
    draw_graph = None

# Repo setup
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils import init_model, setup_logging

_logger = setup_logging(__name__)

# Predefined model families extracted from bin/low_mid_vis/*.sh
CATALOG = {
    "ResNet": [
        "resnet50s.gluon_in1k",
        "resnet101.gluon_in1k",
    ],
    "ResNeXt": [
        "resnext101_32x4d.fb_swsl_ig1b_ft_in1k",
        "resnext101_32x8d.fb_swsl_ig1b_ft_in1k",
    ],
    "ConvNeXt": [
        "convnext_tiny.fb_in1k",
        "convnext_base.fb_in1k",
        "convnext_base.clip_laion2b_augreg_ft_in1k",
        "convnext_large.fb_in1k",
        "convnext_large_mlp.clip_laion2b_augreg_ft_in1k_384",
        "convnext_xlarge.fb_in22k_ft_in1k",
    ],
    "Vision Transformers (ViT)": [
        "vit_base_patch16_clip_224.openai_ft_in12k_in1k",
        "vit_large_patch14_clip_224.openai_ft_in12k_in1k",
        "vit_large_patch14_clip_224.laion2b_ft_in12k_in1k",
    ],
    "DeiT3": [
        "deit3_base_patch16_224.fb_in1k",
        "deit3_medium_patch16_224.fb_in1k",
        "deit3_large_patch16_224.fb_in1k",
        "deit3_large_patch16_224.fb_in22k_ft_in1k",
    ],
    "FocalNet": [
        "focalnet_base_srf.ms_in1k",
        "focalnet_base_lrf.ms_in1k",
    ],
    "Swin Transformer": [
        "swin_base_patch4_window7_224.ms_in1k",
        "swin_s3_base_224.ms_in1k",
        "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
    ],
}

# Key mathematical and stream aggregation operations to include in the graph
SEMANTIC_OPS: Set[str] = {
    "add",
    "add_",
    "scaled_dot_product_attention",
    "matmul",
    "bmm",
    "softmax",
    "cat",
    "concat",
    "unbind",
    "split",
    "mul",
    "mul_",
}


def load_model(
    model_name: str,
    pretrained: bool = False,
    custom_loader: Optional[str] = None,
) -> nn.Module:
    """Load a model by name or custom entrypoint and ensure it is pinned to CPU."""
    if custom_loader:
        if ":" in custom_loader:
            module_path, target = custom_loader.split(":", 1)
            if os.path.exists(module_path):
                spec = importlib.util.spec_from_file_location("custom_loader_mod", module_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                fn = getattr(mod, target)
            else:
                mod = importlib.import_module(module_path)
                fn = getattr(mod, target)
            model = fn(model_name, pretrained=pretrained)
        else:
            raise ValueError(f"Custom loader must be 'path_or_module:function', got '{custom_loader}'")
    else:
        try:
            model = init_model(model_name, pretrained=pretrained)
        except Exception:
            if timm is not None:
                model = timm.create_model(model_name, pretrained=pretrained)
            else:
                raise
    return model.cpu().eval()


def parse_input_size(
    model: nn.Module,
    input_size_str: Optional[str] = None,
) -> Union[Tuple[int, ...], List[Tuple[int, ...]]]:
    """Parse user input size or extract from model's pretrained_cfg."""
    if input_size_str and input_size_str.strip():
        parts = [p.strip() for p in input_size_str.split(";") if p.strip()]
        sizes = []
        for p in parts:
            nums = [
                int(x.strip())
                for x in p.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")
                if x.strip()
            ]
            sizes.append(tuple(nums))
        return sizes[0] if len(sizes) == 1 else sizes

    if hasattr(model, "pretrained_cfg") and isinstance(model.pretrained_cfg, dict):
        cfg_size = model.pretrained_cfg.get("input_size")
        if cfg_size:
            return (1, *cfg_size)

    return (1, 3, 224, 224)


def sanitize_svg(svg_str: str) -> str:
    """Remove hardcoded pt width/height from Graphviz SVG to ensure responsive scaling."""
    def clean_root_svg(match: re.Match) -> str:
        tag = match.group(0)
        tag = re.sub(r'\s+width=["\'][^"\']+["\']', '', tag)
        tag = re.sub(r'\s+height=["\'][^"\']+["\']', '', tag)
        return tag

    return re.sub(r'<svg[^>]+>', clean_root_svg, svg_str, count=1)


def inspect_architecture(
    model: nn.Module,
    input_size: Union[Tuple[int, ...], List[Tuple[int, ...]]],
    depth: Optional[int] = 3,
    direction: str = "LR",
) -> Dict[str, Any]:
    """Extract hierarchy, runtime shapes via hooks, and full computation DAG with functional ops."""
    model = model.cpu().eval()
    device = torch.device("cpu")

    shapes: Dict[str, Dict[str, Any]] = {}
    hooks = []

    def make_hook(name: str):
        def hook(m: nn.Module, inp: Any, out: Any):
            def fmt_shape(t: Any) -> Any:
                if isinstance(t, torch.Tensor):
                    return list(t.shape)
                elif isinstance(t, (tuple, list)):
                    res = [fmt_shape(x) for x in t if x is not None]
                    return res[0] if len(res) == 1 else res
                elif isinstance(t, dict):
                    return {k: fmt_shape(v) for k, v in t.items()}
                return str(type(t).__name__)

            shapes[name] = {
                "in": fmt_shape(inp),
                "out": fmt_shape(out),
            }

        return hook

    for name, mod in model.named_modules():
        hooks.append(mod.register_forward_hook(make_hook(name)))

    # Run forward pass with dummy tensor(s) to capture exact shapes
    try:
        with torch.no_grad():
            if isinstance(input_size, list):
                dummy_inputs = [torch.randn(*s, device=device) for s in input_size]
                model(*dummy_inputs)
            else:
                dummy_input = torch.randn(*input_size, device=device)
                model(dummy_input)
    except Exception as e:
        _logger.warning(f"Dummy forward pass warning: {e}")
    finally:
        for h in hooks:
            h.remove()

    # Build module hierarchy tree
    total_model_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    hierarchy: List[Dict[str, Any]] = []
    mod_id_to_entry: Dict[int, Dict[str, Any]] = {}
    path_to_entry: Dict[str, Dict[str, Any]] = {}

    for path, mod in model.named_modules():
        if not path:
            continue
        parts = path.split(".")
        parent_path = ".".join(parts[:-1]) if len(parts) > 1 else None
        is_leaf = len(list(mod.children())) == 0
        direct_params = sum(p.numel() for p in mod.parameters(recurse=False))
        total_mod_params = sum(p.numel() for p in mod.parameters(recurse=True))

        attrs: Dict[str, Any] = {}
        for attr_name in [
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "in_features",
            "out_features",
            "num_features",
            "normalized_shape",
            "eps",
            "act_layer",
            "drop_rate",
        ]:
            if hasattr(mod, attr_name):
                val = getattr(mod, attr_name)
                attrs[attr_name] = str(val)

        if hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor):
            attrs["weight_shape"] = list(mod.weight.shape)
        if hasattr(mod, "bias") and isinstance(mod.bias, torch.Tensor):
            attrs["bias_shape"] = list(mod.bias.shape)

        sh = shapes.get(path, {})
        in_shape = sh.get("in")
        out_shape = sh.get("out")

        escaped_path = re.escape(path)
        spec_patterns = {
            "out": f"^{escaped_path}:out",
            "in": f"^{escaped_path}:in",
            "delta": f"^{escaped_path}:delta",
            "all": f"^{escaped_path}:all",
        }

        # Classify module category for color coding & badges
        cname = type(mod).__name__
        category = "container"
        if "Conv" in cname:
            category = "conv"
        elif "Linear" in cname or "Dense" in cname:
            category = "linear"
        elif "Norm" in cname:
            category = "norm"
        elif any(act in cname for act in ["ReLU", "GELU", "SiLU", "Sigmoid", "Tanh", "Act"]):
            category = "act"
        elif "Attention" in cname or "SelfAttention" in cname or "Mlp" in cname:
            category = "attention"
        elif "Pool" in cname:
            category = "pool"
        elif is_leaf:
            category = "leaf"

        entry = {
            "id": path,
            "name": parts[-1],
            "parent": parent_path,
            "class_name": cname,
            "category": category,
            "is_leaf": is_leaf,
            "depth": len(parts),
            "direct_params": direct_params,
            "total_params": total_mod_params,
            "param_percent": round((total_mod_params / max(1, total_model_params)) * 100, 2),
            "input_shape": in_shape,
            "output_shape": out_shape,
            "attributes": attrs,
            "spec_patterns": spec_patterns,
            "compute_id": id(mod),
        }
        hierarchy.append(entry)
        mod_id_to_entry[id(mod)] = entry
        path_to_entry[path] = entry

    # Base Cytoscape module nodes
    cytoscape_nodes: List[Dict[str, Any]] = []
    for item in hierarchy:
        cytoscape_nodes.append({
            "data": {
                "id": item["id"],
                "label": f"{item['name']}\\n{item['class_name']}",
                "name": item["name"],
                "parent": item["parent"],
                "class_name": item["class_name"],
                "category": item["category"],
                "is_leaf": item["is_leaf"],
                "depth": item["depth"],
                "in_shape": str(item["input_shape"] or ""),
                "out_shape": str(item["output_shape"] or ""),
                "total_params": item["total_params"],
            }
        })

    valid_node_ids = {n["data"]["id"] for n in cytoscape_nodes}

    # Generate Computation Graph via torchview at full infinite depth
    svg_content = ""
    cytoscape_edges: List[Dict[str, Any]] = []

    if draw_graph is not None:
        try:
            # Full depth trace guarantees all nested blocks, MLPs, attention ops, and residual skips are caught
            cg = draw_graph(
                model,
                input_size=input_size,
                depth=float("inf"),
                device="cpu",
                graph_dir=direction,
                expand_nested=True,
                save_graph=False,
            )

            # Render & Sanitize SVG
            try:
                svg_bytes = cg.visual_graph.pipe(format="svg")
                raw_svg = svg_bytes.decode("utf-8")
                svg_content = sanitize_svg(raw_svg)
            except Exception as e:
                _logger.warning(f"Graphviz SVG pipe error: {e}")
                svg_content = f"<div class='p-4 text-amber-500'>Graphviz SVG render unavailable: {e}</div>"

            # Map FunctionNodes to enclosing parent module paths using cg.node_hierarchy
            def walk_hierarchy(item: Any, current_mod: Optional[str] = None):
                if isinstance(item, dict):
                    for k, v in item.items():
                        mod_path = current_mod
                        if type(k).__name__ == "ModuleNode":
                            cu_id = getattr(k, "compute_unit_id", None)
                            if cu_id in mod_id_to_entry:
                                mod_path = mod_id_to_entry[cu_id]["id"]
                        yield from walk_hierarchy(v, mod_path)
                elif isinstance(item, list):
                    for elem in item:
                        yield from walk_hierarchy(elem, current_mod)
                else:
                    if type(item).__name__ == "FunctionNode":
                        yield (item, current_mod)

            fn_parents: Dict[Any, Optional[str]] = dict(walk_hierarchy(cg.node_hierarchy))

            # Helper for op symbols
            def get_op_symbol(op_name: str) -> str:
                if "add" in op_name:
                    return "+"
                if "matmul" in op_name or "bmm" in op_name:
                    return "×"
                if "attention" in op_name:
                    return "⚡"
                if "softmax" in op_name:
                    return "σ"
                if "cat" in op_name:
                    return "⫴"
                if "unbind" in op_name or "split" in op_name:
                    return "⑂"
                if "mul" in op_name:
                    return "*"
                return "ƒ"

            # Add semantic FunctionNodes as first-class Cytoscape nodes
            added_fn_ids = set()
            for src, dst in cg.edge_list:
                for node in (src, dst):
                    if type(node).__name__ == "FunctionNode" and node.name in SEMANTIC_OPS:
                        fn_id = f"fn_{node.name}_{node.node_id}"
                        if fn_id not in added_fn_ids:
                            added_fn_ids.add(fn_id)
                            parent_path = fn_parents.get(node)
                            # Ensure parent exists in valid_node_ids
                            if parent_path and parent_path not in valid_node_ids:
                                parent_path = None

                            in_s = getattr(node, "input_shape", None)
                            out_s = getattr(node, "output_shape", None)
                            symbol = get_op_symbol(node.name)

                            cytoscape_nodes.append({
                                "data": {
                                    "id": fn_id,
                                    "label": f"{symbol} {node.name}",
                                    "name": node.name,
                                    "parent": parent_path,
                                    "class_name": f"Op:{node.name}",
                                    "category": "op",
                                    "is_leaf": True,
                                    "depth": (len(parent_path.split(".")) + 1) if parent_path else 1,
                                    "in_shape": str(in_s or ""),
                                    "out_shape": str(out_s or ""),
                                    "total_params": 0,
                                }
                            })
                            valid_node_ids.add(fn_id)

            # Build adjacency of all raw edges
            adj = defaultdict(list)
            for src, dst in cg.edge_list:
                adj[src].append(dst)

            def resolve_node(node: Any) -> Optional[str]:
                tname = type(node).__name__
                if tname == "ModuleNode":
                    c_id = getattr(node, "compute_unit_id", None)
                    if c_id and c_id in mod_id_to_entry:
                        return mod_id_to_entry[c_id]["id"]
                elif tname == "FunctionNode" and node.name in SEMANTIC_OPS:
                    return f"fn_{node.name}_{node.node_id}"
                return None

            visited_edges = set()
            for src, _ in cg.edge_list:
                src_id = resolve_node(src)
                if not src_id:
                    continue

                queue = list(adj[src])
                seen = set()
                while queue:
                    curr = queue.pop(0)
                    if curr in seen:
                        continue
                    seen.add(curr)

                    dst_id = resolve_node(curr)
                    if dst_id:
                        if src_id != dst_id and src_id in valid_node_ids and dst_id in valid_node_ids:
                            edge_key = (src_id, dst_id)
                            if edge_key not in visited_edges:
                                visited_edges.add(edge_key)
                                out_s = getattr(src, "output_shape", None)
                                in_s = getattr(curr, "input_shape", None)
                                shape_lbl = ""
                                if out_s and isinstance(out_s, list) and len(out_s) > 0:
                                    shape_lbl = str(out_s[0])
                                elif in_s and isinstance(in_s, list) and len(in_s) > 0:
                                    shape_lbl = str(in_s[0])

                                cytoscape_edges.append({
                                    "data": {
                                        "id": f"e_{src_id}_to_{dst_id}".replace(".", "_"),
                                        "source": src_id,
                                        "target": dst_id,
                                        "shape": shape_lbl,
                                    }
                                })
                        # Stop traversing past this recognized node
                    else:
                        # Continue traversing through intermediate auxiliary tensors / hidden functions
                        queue.extend(adj[curr])

        except Exception as e:
            _logger.error(f"Error extracting computation graph: {e}", exc_info=True)

    # Enforce strict edge validation guarantee
    cytoscape_edges = [
        e for e in cytoscape_edges
        if e["data"]["source"] in valid_node_ids and e["data"]["target"] in valid_node_ids
    ]

    return {
        "model_info": {
            "name": getattr(model, "default_cfg", {}).get("architecture", type(model).__name__),
            "class": type(model).__name__,
            "total_params": total_model_params,
            "trainable_params": trainable_params,
            "input_size": input_size,
            "depth": depth,
            "total_modules": len(hierarchy),
            "leaf_modules": sum(1 for m in hierarchy if m["is_leaf"]),
        },
        "hierarchy": hierarchy,
        "svg": svg_content,
        "cytoscape": {
            "nodes": cytoscape_nodes,
            "edges": cytoscape_edges,
        },
    }


# ==============================================================================
# Web Application Frontend (HTML / CSS / JS)
# ==============================================================================
HTML_PAGE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Model Architecture Explorer</title>
  <!-- Tailwind CSS -->
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: {
              50: '#eef2ff',
              500: '#6366f1',
              600: '#4f46e5',
              700: '#4338ca',
            },
            surface: {
              900: '#0f172a',
              800: '#1e293b',
              700: '#334155',
              600: '#475569',
            }
          }
        }
      }
    }
  </script>
  <!-- Cytoscape.js & Dagre -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.28.1/cytoscape.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
  <!-- SVG Pan Zoom -->
  <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
  <style>
    /* Custom scrollbars */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }

    #svg-container svg {
      width: 100% !important;
      height: 100% !important;
      max-width: none !important;
      display: block;
    }
    .node-highlight polygon, .node-highlight path, .node-highlight rect {
      stroke: #38bdf8 !important;
      stroke-width: 3px !important;
      filter: drop-shadow(0 0 8px #38bdf8);
    }
  </style>
</head>
<body class="bg-surface-900 text-slate-100 h-screen flex flex-col overflow-hidden font-sans">

  <!-- Top Navigation Header -->
  <header class="bg-surface-800 border-b border-surface-700 px-4 py-2.5 flex items-center justify-between shrink-0 shadow-md">
    <div class="flex items-center gap-3">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center font-bold text-white shadow">
        M
      </div>
      <div>
        <h1 class="font-bold text-sm tracking-wide text-white">Model Architecture Explorer</h1>
        <p class="text-xs text-slate-400">Interactive Hierarchy, Connections & Shapes</p>
      </div>
    </div>

    <!-- Controls Toolbar -->
    <div class="flex items-center gap-3">
      <!-- Preset Model Dropdown -->
      <div class="relative">
        <select id="preset-select" class="bg-surface-900 border border-surface-600 rounded-md px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:ring-1 focus:ring-brand-500 max-w-[210px]">
          <option value="">-- Presets (bin/low_mid_vis) --</option>
        </select>
      </div>

      <!-- Custom Model Input -->
      <div class="relative">
        <input id="model-input" type="text" placeholder="timm model (e.g. vit_base_patch16_clip_224)"
          class="bg-surface-900 border border-surface-600 rounded-md px-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 w-52 focus:outline-none focus:ring-1 focus:ring-brand-500">
      </div>

      <!-- Input Shape -->
      <div class="flex items-center gap-1.5">
        <span class="text-xs text-slate-400">Shape:</span>
        <input id="shape-input" type="text" value="1, 3, 224, 224" placeholder="1, 3, 224, 224"
          class="bg-surface-900 border border-surface-600 rounded-md px-2.5 py-1.5 text-xs text-slate-200 w-28 focus:outline-none focus:ring-1 focus:ring-brand-500">
      </div>

      <!-- Direction Toggle -->
      <div class="flex items-center gap-1.5">
        <span class="text-xs text-slate-400">Dir:</span>
        <select id="dir-select" class="bg-surface-900 border border-surface-600 rounded-md px-2 py-1.5 text-xs text-slate-200 focus:outline-none">
          <option value="LR" selected>Left &rarr; Right (LR)</option>
          <option value="TB">Top &rarr; Down (TB)</option>
        </select>
      </div>

      <!-- Show Operations Toggle -->
      <label class="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer bg-surface-900/60 border border-surface-700 px-2.5 py-1.5 rounded-md hover:border-slate-500 transition">
        <input id="show-ops-toggle" type="checkbox" checked class="accent-brand-500 rounded">
        <span>Show Ops (add, matmul)</span>
      </label>

      <!-- Depth Slider (UI filtering) -->
      <div class="flex items-center gap-2 bg-surface-900/60 border border-surface-700 rounded-md px-2.5 py-1">
        <span class="text-xs text-slate-400">Depth: <span id="depth-val" class="font-mono text-indigo-400 font-bold">All</span></span>
        <input id="depth-slider" type="range" min="1" max="6" value="6" class="w-14 accent-indigo-500 cursor-pointer">
      </div>

      <!-- Inspect Button -->
      <button id="inspect-btn" class="bg-brand-600 hover:bg-brand-700 active:scale-95 text-white font-medium px-4 py-1.5 rounded-md text-xs transition flex items-center gap-2 shadow">
        <span id="btn-spinner" class="hidden animate-spin">&#9696;</span>
        <span>Inspect</span>
      </button>
    </div>

    <!-- Status & Info Badges -->
    <div class="flex items-center gap-2 text-xs">
      <div id="model-badge" class="hidden px-2 py-1 rounded bg-indigo-950 text-indigo-300 border border-indigo-800 font-mono text-[11px]">
        Loading...
      </div>
      <button id="export-svg-btn" title="Export SVG" class="px-2 py-1 bg-surface-700 hover:bg-surface-600 text-slate-200 rounded text-xs">SVG</button>
      <button id="export-json-btn" title="Export JSON" class="px-2 py-1 bg-surface-700 hover:bg-surface-600 text-slate-200 rounded text-xs">JSON</button>
    </div>
  </header>

  <!-- Sub-header Navigation Tabs & Filter Bar -->
  <div class="bg-surface-800/80 border-b border-surface-700 px-4 py-1.5 flex items-center justify-between shrink-0">
    <div class="flex items-center gap-2">
      <span class="text-xs text-slate-400 font-medium">View:</span>
      <div class="flex bg-surface-900 p-0.5 rounded-lg border border-surface-700 text-xs">
        <button id="tab-cyto" class="view-tab active px-3 py-1 rounded-md font-medium transition bg-brand-600 text-white">Compound Graph</button>
        <button id="tab-svg" class="view-tab px-3 py-1 rounded-md font-medium transition text-slate-400 hover:text-slate-200">Execution DAG (SVG)</button>
        <button id="tab-tree" class="view-tab px-3 py-1 rounded-md font-medium transition text-slate-400 hover:text-slate-200">Hierarchy Table</button>
      </div>
    </div>

    <!-- Search & Regex Highlight -->
    <div class="flex items-center gap-2">
      <div class="relative flex items-center">
        <input id="filter-input" type="text" placeholder="Highlight layers / ops (e.g. add, matmul, qkv)..."
          class="bg-surface-900 border border-surface-600 rounded-md pl-3 pr-8 py-1 text-xs text-slate-200 w-72 focus:outline-none focus:ring-1 focus:ring-sky-500">
        <span id="filter-count" class="absolute right-2 text-[10px] text-slate-400"></span>
      </div>
      <button id="zoom-fit-btn" class="px-2.5 py-1 bg-surface-700 hover:bg-surface-600 text-slate-200 rounded text-xs" title="Reset Zoom / Fit">Fit View</button>
    </div>
  </div>

  <!-- Main View Area (Center Canvas + Right Sidebar) -->
  <div class="flex-1 flex overflow-hidden relative">

    <!-- Center Canvas -->
    <div class="flex-1 flex flex-col relative bg-surface-900 overflow-hidden">
      <!-- Loading Overlay -->
      <div id="loading-overlay" class="hidden absolute inset-0 bg-surface-900/80 backdrop-blur-sm z-50 flex flex-col items-center justify-center gap-3">
        <div class="w-10 h-10 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
        <p id="loading-msg" class="text-sm font-medium text-slate-300">Extracting model hierarchy and tracing shapes...</p>
      </div>

      <!-- View 1: Cytoscape Compound Graph (Default) -->
      <div id="view-cyto" class="view-panel w-full h-full relative">
        <div id="cy" class="w-full h-full"></div>
        <div class="absolute bottom-3 left-3 bg-surface-800/90 border border-surface-700 rounded-lg p-2.5 text-xs text-slate-400 shadow-lg pointer-events-none">
          <div class="font-medium text-slate-200 mb-1">Compound Graph Legend:</div>
          <div class="flex items-center gap-2 flex-wrap">
            <span class="inline-block w-3 h-3 rounded bg-blue-700 border border-blue-400"></span> Conv2d
            <span class="inline-block w-3 h-3 rounded bg-emerald-600 border border-emerald-400"></span> Linear
            <span class="inline-block w-3 h-3 rounded bg-amber-600 border border-amber-400"></span> Norm
            <span class="inline-block w-3 h-3 rounded bg-pink-700 border border-pink-400"></span> Attention
            <span class="inline-block w-3 h-3 rounded-full bg-amber-800 border border-amber-400"></span> + Add
            <span class="inline-block w-3 h-3 rounded bg-emerald-800 border border-emerald-400"></span> × MatMul / SDPA
            <span class="inline-block w-3 h-3 rounded bg-purple-800 border border-purple-400"></span> σ Softmax / Split
          </div>
        </div>
      </div>

      <!-- View 2: Interactive SVG DAG -->
      <div id="view-svg" class="view-panel hidden w-full h-full flex items-center justify-center overflow-hidden">
        <div id="svg-container" class="w-full h-full flex items-center justify-center overflow-hidden">
          <div class="text-slate-500 text-sm">Select a model and click "Inspect" to view architecture.</div>
        </div>
      </div>

      <!-- View 3: Hierarchy Tree Table -->
      <div id="view-tree" class="view-panel hidden w-full h-full overflow-auto p-4">
        <div class="bg-surface-800 border border-surface-700 rounded-lg overflow-hidden shadow">
          <table class="w-full text-left text-xs border-collapse">
            <thead class="bg-surface-700/50 text-slate-300 font-semibold border-b border-surface-600">
              <tr>
                <th class="p-2.5">Module Path</th>
                <th class="p-2.5">Class</th>
                <th class="p-2.5">Input Shape</th>
                <th class="p-2.5">Output Shape</th>
                <th class="p-2.5 text-right">Parameters</th>
                <th class="p-2.5 text-center">Actions</th>
              </tr>
            </thead>
            <tbody id="tree-table-body" class="divide-y divide-surface-700 text-slate-200">
              <!-- Dynamically populated -->
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Right Inspector Sidebar -->
    <div id="sidebar" class="w-80 bg-surface-800 border-l border-surface-700 flex flex-col shrink-0 overflow-y-auto">
      <div class="p-3 border-b border-surface-700 flex items-center justify-between">
        <h2 class="text-xs font-bold uppercase tracking-wider text-slate-400">Module Inspector</h2>
        <span id="selected-type-badge" class="px-2 py-0.5 rounded text-[10px] font-mono bg-surface-700 text-slate-300">None selected</span>
      </div>

      <div id="inspector-content" class="p-4 space-y-4 text-xs">
        <div class="text-slate-500 text-center py-10">Click on any layer or block in the graph to inspect details and copy recording targets.</div>
      </div>
    </div>

  </div>

  <!-- Notification Toast -->
  <div id="toast" class="fixed bottom-5 right-5 bg-brand-600 text-white px-3.5 py-2 rounded-lg text-xs shadow-xl opacity-0 transition-opacity pointer-events-none z-50">
    Copied to clipboard!
  </div>

  <script>
    // Global State
    let currentData = null;
    let cyInstance = null;
    let panZoomInstance = null;
    let selectedModuleId = null;

    // DOM Elements
    const presetSelect = document.getElementById('preset-select');
    const modelInput = document.getElementById('model-input');
    const shapeInput = document.getElementById('shape-input');
    const dirSelect = document.getElementById('dir-select');
    const showOpsToggle = document.getElementById('show-ops-toggle');
    const depthSlider = document.getElementById('depth-slider');
    const depthVal = document.getElementById('depth-val');
    const inspectBtn = document.getElementById('inspect-btn');
    const btnSpinner = document.getElementById('btn-spinner');
    const modelBadge = document.getElementById('model-badge');
    const filterInput = document.getElementById('filter-input');
    const filterCount = document.getElementById('filter-count');
    const zoomFitBtn = document.getElementById('zoom-fit-btn');
    const inspectorContent = document.getElementById('inspector-content');
    const selectedTypeBadge = document.getElementById('selected-type-badge');
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingMsg = document.getElementById('loading-msg');
    const toast = document.getElementById('toast');

    // Tab buttons & panels
    const tabCyto = document.getElementById('tab-cyto');
    const tabSvg = document.getElementById('tab-svg');
    const tabTree = document.getElementById('tab-tree');
    const viewCyto = document.getElementById('view-cyto');
    const viewSvg = document.getElementById('view-svg');
    const viewTree = document.getElementById('view-tree');

    // Show toast message
    function showToast(msg) {
      toast.textContent = msg;
      toast.classList.remove('opacity-0');
      toast.classList.add('opacity-100');
      setTimeout(() => {
        toast.classList.remove('opacity-100');
        toast.classList.add('opacity-0');
      }, 2000);
    }

    // Load Catalog presets
    async function loadCatalog() {
      try {
        const res = await fetch('/api/catalog');
        const catalog = await res.json();
        presetSelect.innerHTML = '<option value="">-- Presets (bin/low_mid_vis) --</option>';
        for (const [family, models] of Object.entries(catalog)) {
          const group = document.createElement('optgroup');
          group.label = family;
          models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            group.appendChild(opt);
          });
          presetSelect.appendChild(group);
        }
      } catch (e) {
        console.error('Failed to load catalog', e);
      }
    }

    presetSelect.addEventListener('change', () => {
      if (presetSelect.value) {
        modelInput.value = presetSelect.value;
      }
    });

    depthSlider.addEventListener('input', () => {
      depthVal.textContent = depthSlider.value >= 6 ? 'All' : depthSlider.value;
    });

    // Direction Change Listener
    dirSelect.addEventListener('change', () => {
      if (cyInstance) {
        runDagreLayout();
      }
    });

    // Show Ops Toggle Listener
    showOpsToggle.addEventListener('change', () => {
      if (cyInstance) {
        const show = showOpsToggle.checked;
        cyInstance.nodes('node[category="op"]').style('display', show ? 'element' : 'none');
        runDagreLayout();
      }
    });

    // Tab Switching
    function setView(tab) {
      [tabCyto, tabSvg, tabTree].forEach(t => {
        t.classList.remove('bg-brand-600', 'text-white');
        t.classList.add('text-slate-400');
      });
      [viewCyto, viewSvg, viewTree].forEach(v => v.classList.add('hidden'));

      if (tab === 'cyto') {
        tabCyto.classList.add('bg-brand-600', 'text-white');
        tabCyto.classList.remove('text-slate-400');
        viewCyto.classList.remove('hidden');
        if (cyInstance) {
          try {
            cyInstance.resize();
            cyInstance.fit(null, 40);
          } catch (e) {
            console.warn('Cytoscape resize error:', e);
          }
        }
      } else if (tab === 'svg') {
        tabSvg.classList.add('bg-brand-600', 'text-white');
        tabSvg.classList.remove('text-slate-400');
        viewSvg.classList.remove('hidden');
        if (panZoomInstance) {
          try {
            panZoomInstance.resize();
            panZoomInstance.fit();
            panZoomInstance.center();
          } catch (e) {
            console.warn('SVG resize error:', e);
          }
        }
      } else if (tab === 'tree') {
        tabTree.classList.add('bg-brand-600', 'text-white');
        tabTree.classList.remove('text-slate-400');
        viewTree.classList.remove('hidden');
      }
    }

    tabCyto.addEventListener('click', () => setView('cyto'));
    tabSvg.addEventListener('click', () => setView('svg'));
    tabTree.addEventListener('click', () => setView('tree'));

    // Inspect Model Action
    async function triggerInspect() {
      const modelName = modelInput.value.trim();
      if (!modelName) {
        alert('Please enter or select a model name.');
        return;
      }

      btnSpinner.classList.remove('hidden');
      inspectBtn.disabled = true;
      loadingOverlay.classList.remove('hidden');
      loadingMsg.textContent = `Loading <${modelName}> and tracing computation graph with functional operations...`;

      try {
        const payload = {
          model_name: modelName,
          input_size: shapeInput.value.trim(),
          depth: parseInt(depthSlider.value),
          direction: dirSelect.value
        };

        const res = await fetch('/api/inspect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.error || 'Server inspection failed');
        }

        currentData = await res.json();
        renderAllViews(currentData);

      } catch (err) {
        alert(`Inspection Error: ${err.message}`);
        console.error(err);
      } finally {
        btnSpinner.classList.add('hidden');
        inspectBtn.disabled = false;
        loadingOverlay.classList.add('hidden');
      }
    }

    inspectBtn.addEventListener('click', triggerInspect);
    modelInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') triggerInspect();
    });

    // Render All Views
    function renderAllViews(data) {
      // 1. Update Header Badges
      const info = data.model_info;
      modelBadge.classList.remove('hidden');
      modelBadge.textContent = `${info.name} (${(info.total_params / 1e6).toFixed(2)}M params, ${info.total_modules} modules)`;

      // 2. Render Cytoscape Graph (Default)
      renderCytoscape(data.cytoscape);

      // 3. Render SVG View
      renderSvg(data.svg);

      // 4. Render Hierarchy Tree Table
      renderTreeTable(data.hierarchy);

      // 5. Select first module if available
      if (data.hierarchy && data.hierarchy.length > 0) {
        selectModule(data.hierarchy[0].id);
      }
    }

    // Run Dagre Layout
    function runDagreLayout() {
      if (!cyInstance) return;
      cyInstance.layout({
        name: 'dagre',
        rankDir: dirSelect.value,
        nodeSep: 50,
        rankSep: 85,
        padding: 40
      }).run();
    }

    // Render Cytoscape Graph
    function renderCytoscape(cyData) {
      if (cyInstance) {
        try {
          cyInstance.destroy();
        } catch (e) {
          console.warn('Destroy cyInstance warning:', e);
        }
        cyInstance = null;
      }

      try {
        cyInstance = cytoscape({
          container: document.getElementById('cy'),
          elements: {
            nodes: cyData.nodes || [],
            edges: cyData.edges || []
          },
          layout: {
            name: 'dagre',
            rankDir: dirSelect.value,
            nodeSep: 50,
            rankSep: 85,
            padding: 40
          },
          style: [
            {
              selector: 'node',
              style: {
                'label': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'font-size': '10px',
                'color': '#f8fafc',
                'background-color': '#1e293b',
                'border-width': 1.5,
                'border-color': '#475569',
                'shape': 'roundrectangle',
                'padding': '8px',
                'text-wrap': 'wrap'
              }
            },
            {
              selector: 'node:parent',
              style: {
                'background-color': '#0f172a',
                'background-opacity': 0.45,
                'border-color': '#4f46e5',
                'border-width': 2,
                'text-valign': 'top',
                'text-halign': 'center',
                'font-size': '11px',
                'font-weight': 'bold',
                'color': '#818cf8',
                'padding': '16px'
              }
            },
            {
              selector: 'node[category="conv"]',
              style: { 'background-color': '#1e3a8a', 'border-color': '#3b82f6' }
            },
            {
              selector: 'node[category="linear"]',
              style: { 'background-color': '#064e3b', 'border-color': '#10b981' }
            },
            {
              selector: 'node[category="norm"]',
              style: { 'background-color': '#78350f', 'border-color': '#f59e0b' }
            },
            {
              selector: 'node[category="act"]',
              style: { 'background-color': '#581c87', 'border-color': '#a855f7' }
            },
            {
              selector: 'node[category="attention"]',
              style: { 'background-color': '#831843', 'border-color': '#ec4899' }
            },
            {
              selector: 'node[category="op"]',
              style: {
                'background-color': '#1e293b',
                'border-color': '#f59e0b',
                'border-width': 2,
                'shape': 'roundrectangle',
                'color': '#fbbf24',
                'font-size': '9px',
                'font-weight': 'bold',
                'padding': '5px'
              }
            },
            {
              selector: 'node[name="add"], node[name="add_"]',
              style: {
                'background-color': '#78350f',
                'border-color': '#f59e0b',
                'shape': 'ellipse',
                'width': '34px',
                'height': '34px',
                'color': '#fde68a'
              }
            },
            {
              selector: 'node[name="scaled_dot_product_attention"], node[name="matmul"], node[name="bmm"]',
              style: {
                'background-color': '#064e3b',
                'border-color': '#10b981',
                'shape': 'roundrectangle',
                'color': '#a7f3d0'
              }
            },
            {
              selector: 'node[name="softmax"]',
              style: {
                'background-color': '#581c87',
                'border-color': '#a855f7',
                'shape': 'roundrectangle',
                'color': '#f3e8ff'
              }
            },
            {
              selector: 'node[name="cat"], node[name="concat"], node[name="unbind"], node[name="split"]',
              style: {
                'background-color': '#1f2937',
                'border-color': '#9ca3af',
                'shape': 'roundrectangle',
                'color': '#e5e7eb'
              }
            },
            {
              selector: 'node.selected',
              style: {
                'border-color': '#38bdf8',
                'border-width': 3,
                'shadow-blur': 12,
                'shadow-color': '#38bdf8',
                'shadow-opacity': 0.8
              }
            },
            {
              selector: 'edge',
              style: {
                'width': 1.8,
                'line-color': '#64748b',
                'target-arrow-color': '#64748b',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'control-point-step-size': 45,
                'arrow-scale': 0.85,
                'label': 'data(shape)',
                'font-size': '8px',
                'color': '#94a3b8',
                'text-background-color': '#0f172a',
                'text-background-opacity': 0.8,
                'text-background-padding': '2px',
                'text-rotation': 'autorotate'
              }
            }
          ]
        });

        // Apply visibility from toggle
        if (!showOpsToggle.checked) {
          cyInstance.nodes('node[category="op"]').style('display', 'none');
          runDagreLayout();
        }

        cyInstance.on('tap', 'node', (e) => {
          const node = e.target;
          selectModule(node.id());
        });

      } catch (err) {
        console.error('Cytoscape render error:', err);
        document.getElementById('cy').innerHTML = `<div class="p-6 text-amber-400 text-xs">Compound graph render notice: ${err.message}. Switch to Execution DAG or Hierarchy Table.</div>`;
      }
    }

    // Render SVG Flow
    function renderSvg(svgHtml) {
      const container = document.getElementById('svg-container');
      if (panZoomInstance) {
        try {
          panZoomInstance.destroy();
        } catch (e) {
          console.warn('Destroy panZoom warning:', e);
        }
        panZoomInstance = null;
      }

      container.innerHTML = svgHtml || '<div class="text-slate-500">No SVG available</div>';

      const svgElem = container.querySelector('svg');
      if (!svgElem) return;

      svgElem.style.width = '100%';
      svgElem.style.height = '100%';
      svgElem.style.minHeight = '300px';

      // Add click events on SVG nodes
      const nodes = svgElem.querySelectorAll('.node, .cluster');
      nodes.forEach(n => {
        n.style.cursor = 'pointer';
        n.addEventListener('click', (e) => {
          e.stopPropagation();
          const title = n.querySelector('title');
          const titleText = title ? title.textContent.trim() : '';
          if (!currentData || !currentData.hierarchy) return;
          const match = currentData.hierarchy.find(m =>
            m.id === titleText || m.name === titleText || titleText.includes(m.id)
          );
          if (match) {
            selectModule(match.id);
          }
        });
      });

      requestAnimationFrame(() => {
        try {
          const rect = svgElem.getBoundingClientRect();
          if (rect.width > 0 && rect.height > 0 && typeof svgPanZoom === 'function') {
            panZoomInstance = svgPanZoom(svgElem, {
              zoomEnabled: true,
              controlIconsEnabled: false,
              fit: true,
              center: true,
              minZoom: 0.05,
              maxZoom: 30,
              dblClickZoomEnabled: true
            });
          }
        } catch (err) {
          console.warn('svgPanZoom fallback active:', err);
          container.style.overflow = 'auto';
        }
      });
    }

    // Render Hierarchy Table
    function renderTreeTable(hierarchy) {
      const tbody = document.getElementById('tree-table-body');
      tbody.innerHTML = '';

      hierarchy.forEach(item => {
        const tr = document.createElement('tr');
        tr.id = `tree-row-${item.id}`;
        tr.className = 'hover:bg-surface-700/40 cursor-pointer transition';

        const indent = (item.depth - 1) * 16;
        const inStr = item.input_shape ? JSON.stringify(item.input_shape) : '-';
        const outStr = item.output_shape ? JSON.stringify(item.output_shape) : '-';
        const paramStr = item.total_params > 0 ? item.total_params.toLocaleString() : '-';

        tr.innerHTML = `
          <td class="p-2.5 font-mono text-[11px]" style="padding-left: ${indent + 10}px">
            <span class="text-slate-400">${item.parent ? '↳ ' : ''}</span>
            <span class="font-semibold text-slate-100">${item.name}</span>
          </td>
          <td class="p-2.5">
            <span class="px-1.5 py-0.5 rounded text-[10px] font-mono bg-surface-700 text-slate-300">${item.class_name}</span>
          </td>
          <td class="p-2.5 font-mono text-[11px] text-slate-300">${inStr}</td>
          <td class="p-2.5 font-mono text-[11px] text-slate-300">${outStr}</td>
          <td class="p-2.5 text-right font-mono text-[11px] text-slate-400">${paramStr}</td>
          <td class="p-2.5 text-center">
            <button class="copy-spec-btn px-2 py-0.5 bg-surface-600 hover:bg-brand-600 text-slate-200 rounded text-[10px]" data-spec="${item.spec_patterns.out}">
              Copy Spec
            </button>
          </td>
        `;

        tr.addEventListener('click', () => selectModule(item.id));
        tbody.appendChild(tr);
      });

      document.querySelectorAll('.copy-spec-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          navigator.clipboard.writeText(btn.dataset.spec);
          showToast(`Copied: ${btn.dataset.spec}`);
        });
      });
    }

    // Select & Inspect Module or Operation
    function selectModule(moduleId) {
      if (!currentData) return;
      selectedModuleId = moduleId;

      // Check if it's a module
      const item = currentData.hierarchy.find(m => m.id === moduleId);
      
      // Check if it's a function op
      const opNode = (!item && currentData.cytoscape) ? currentData.cytoscape.nodes.find(n => n.data.id === moduleId) : null;

      if (!item && !opNode) return;

      const title = item ? item.id : opNode.data.id;
      const cname = item ? item.class_name : opNode.data.class_name;
      selectedTypeBadge.textContent = cname;

      if (cyInstance) {
        try {
          cyInstance.nodes().removeClass('selected');
          const cyNode = cyInstance.$(`#${CSS.escape(moduleId)}`);
          if (cyNode) cyNode.addClass('selected');
        } catch (e) {}
      }

      document.querySelectorAll('#tree-table-body tr').forEach(r => r.classList.remove('bg-surface-700/60'));
      const activeRow = document.getElementById(`tree-row-${moduleId}`);
      if (activeRow) {
        activeRow.classList.add('bg-surface-700/60');
      }

      const container = document.getElementById('svg-container');
      const allSvgNodes = container.querySelectorAll('.node, .cluster');
      allSvgNodes.forEach(n => n.classList.remove('node-highlight'));
      allSvgNodes.forEach(n => {
        const t = n.querySelector('title');
        if (t && (t.textContent.trim() === moduleId || (item && t.textContent.trim() === item.name))) {
          n.classList.add('node-highlight');
        }
      });

      if (opNode && !item) {
        // Operation node inspection
        inspectorContent.innerHTML = `
          <div class="space-y-1">
            <div class="text-[11px] text-amber-400 uppercase font-semibold">Functional Tensor Operation</div>
            <div class="font-mono text-sm font-bold text-slate-100">${opNode.data.name}</div>
          </div>
          <div class="bg-surface-900/60 p-2.5 rounded border border-surface-700 space-y-1.5 text-[11px]">
            <div class="flex justify-between">
              <span class="text-slate-400">Operation Type:</span>
              <span class="font-mono font-bold text-amber-400">${opNode.data.name}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-400">Parent Module:</span>
              <span class="font-mono text-indigo-400">${opNode.data.parent || 'Root'}</span>
            </div>
          </div>
          <div class="space-y-1">
            <div class="text-[11px] text-slate-400 uppercase font-semibold">Tensor Dimensions</div>
            <div class="bg-surface-900/60 p-2.5 rounded border border-surface-700 font-mono text-[11px] space-y-1">
              <div><span class="text-emerald-400 font-bold">Input:</span> <span class="text-slate-200">${opNode.data.in_shape || 'N/A'}</span></div>
              <div><span class="text-sky-400 font-bold">Output:</span> <span class="text-slate-200">${opNode.data.out_shape || 'N/A'}</span></div>
            </div>
          </div>
          <p class="text-[10px] text-slate-400 bg-surface-900/40 p-2 rounded border border-surface-700">
            This node represents a stream convergence/divergence point (e.g. residual addition, attention matrix multiplication, or softmax).
          </p>
        `;
        return;
      }

      // Module node inspection
      const inStr = item.input_shape ? JSON.stringify(item.input_shape, null, 2) : 'N/A';
      const outStr = item.output_shape ? JSON.stringify(item.output_shape, null, 2) : 'N/A';

      let attrsHtml = '';
      if (item.attributes && Object.keys(item.attributes).length > 0) {
        attrsHtml = `
          <div class="space-y-1 bg-surface-900/60 p-2.5 rounded border border-surface-700 font-mono text-[11px]">
            ${Object.entries(item.attributes).map(([k, v]) => `
              <div class="flex justify-between">
                <span class="text-slate-400">${k}:</span>
                <span class="text-slate-200">${v}</span>
              </div>
            `).join('')}
          </div>
        `;
      }

      inspectorContent.innerHTML = `
        <div class="space-y-1">
          <div class="text-[11px] text-slate-400 uppercase font-semibold">Module Path</div>
          <div class="font-mono text-sm font-bold text-indigo-400 break-all">${item.id}</div>
        </div>

        <div class="grid grid-cols-2 gap-2 text-[11px]">
          <div class="bg-surface-900/60 p-2 rounded border border-surface-700">
            <span class="text-slate-400 block">Class:</span>
            <span class="font-mono font-semibold text-slate-200">${item.class_name}</span>
          </div>
          <div class="bg-surface-900/60 p-2 rounded border border-surface-700">
            <span class="text-slate-400 block">Hierarchy Depth:</span>
            <span class="font-mono font-semibold text-slate-200">${item.depth}</span>
          </div>
        </div>

        <div class="space-y-1">
          <div class="text-[11px] text-slate-400 uppercase font-semibold">Tensor Shapes</div>
          <div class="space-y-2 bg-surface-900/60 p-2.5 rounded border border-surface-700 font-mono text-[11px]">
            <div>
              <span class="text-emerald-400 font-bold">Input:</span>
              <pre class="text-slate-200 mt-0.5 whitespace-pre-wrap">${inStr}</pre>
            </div>
            <div>
              <span class="text-sky-400 font-bold">Output:</span>
              <pre class="text-slate-200 mt-0.5 whitespace-pre-wrap">${outStr}</pre>
            </div>
          </div>
        </div>

        <div class="space-y-1">
          <div class="text-[11px] text-slate-400 uppercase font-semibold">Parameters</div>
          <div class="bg-surface-900/60 p-2.5 rounded border border-surface-700 space-y-1 text-[11px]">
            <div class="flex justify-between">
              <span class="text-slate-400">Total (sub-tree):</span>
              <span class="font-mono font-bold text-slate-200">${item.total_params.toLocaleString()}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-400">Direct (layer):</span>
              <span class="font-mono text-slate-300">${item.direct_params.toLocaleString()}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-slate-400">% of Model:</span>
              <span class="font-mono text-indigo-400">${item.param_percent}%</span>
            </div>
          </div>
        </div>

        ${attrsHtml ? `
        <div class="space-y-1">
          <div class="text-[11px] text-slate-400 uppercase font-semibold">Layer Attributes</div>
          ${attrsHtml}
        </div>
        ` : ''}

        <div class="space-y-1.5 pt-2 border-t border-surface-700">
          <div class="text-[11px] text-slate-300 uppercase font-bold flex items-center justify-between">
            <span>Recording Targets</span>
            <span class="text-[10px] text-slate-400 lowercase">(src/layer_spec.py)</span>
          </div>
          <p class="text-[10px] text-slate-400">Click below to copy regex patterns ready for evaluation scripts:</p>
          <div class="grid grid-cols-2 gap-1.5">
            <button class="spec-btn bg-surface-700 hover:bg-brand-600 text-slate-200 p-1.5 rounded text-[11px] font-mono text-left transition" data-val="${item.spec_patterns.out}">
              :out stream 📋
            </button>
            <button class="spec-btn bg-surface-700 hover:bg-brand-600 text-slate-200 p-1.5 rounded text-[11px] font-mono text-left transition" data-val="${item.spec_patterns.in}">
              :in stream 📋
            </button>
            <button class="spec-btn bg-surface-700 hover:bg-brand-600 text-slate-200 p-1.5 rounded text-[11px] font-mono text-left transition" data-val="${item.spec_patterns.delta}">
              :delta stream 📋
            </button>
            <button class="spec-btn bg-surface-700 hover:bg-brand-600 text-slate-200 p-1.5 rounded text-[11px] font-mono text-left transition" data-val="${item.spec_patterns.all}">
              :all streams 📋
            </button>
          </div>
        </div>
      `;

      document.querySelectorAll('.spec-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          navigator.clipboard.writeText(btn.dataset.val);
          showToast(`Copied: ${btn.dataset.val}`);
        });
      });
    }

    // Filter & Search
    filterInput.addEventListener('input', () => {
      const q = filterInput.value.trim();
      if (!currentData) return;

      if (!q) {
        filterCount.textContent = '';
        if (cyInstance) cyInstance.nodes().removeClass('selected');
        return;
      }

      let count = 0;
      let regex = null;
      try {
        regex = new RegExp(q, 'i');
      } catch (e) {
        regex = null;
      }

      currentData.hierarchy.forEach(item => {
        const match = regex ? (regex.test(item.id) || regex.test(item.class_name)) : item.id.toLowerCase().includes(q.toLowerCase());
        if (match) count++;

        const row = document.getElementById(`tree-row-${item.id}`);
        if (row) {
          row.style.display = match ? '' : 'none';
        }
      });

      filterCount.textContent = `${count} matches`;

      if (cyInstance) {
        cyInstance.nodes().forEach(node => {
          const match = regex ? (regex.test(node.id()) || regex.test(node.data('class_name'))) : node.id().toLowerCase().includes(q.toLowerCase());
          if (match) {
            node.addClass('selected');
          } else {
            node.removeClass('selected');
          }
        });
      }
    });

    // Zoom Fit Button
    zoomFitBtn.addEventListener('click', () => {
      if (cyInstance) {
        try {
          cyInstance.fit(null, 40);
        } catch (e) {}
      }
      if (panZoomInstance) {
        try {
          panZoomInstance.reset();
          panZoomInstance.fit();
          panZoomInstance.center();
        } catch (e) {}
      }
    });

    // Export Handlers
    document.getElementById('export-svg-btn').addEventListener('click', () => {
      if (!currentData || !currentData.svg) {
        alert('No SVG available to export');
        return;
      }
      const blob = new Blob([currentData.svg], { type: 'image/svg+xml' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${currentData.model_info.name || 'model'}_architecture.svg`;
      a.click();
      URL.revokeObjectURL(url);
    });

    document.getElementById('export-json-btn').addEventListener('click', () => {
      if (!currentData) {
        alert('No data to export');
        return;
      }
      const blob = new Blob([JSON.stringify(currentData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${currentData.model_info.name || 'model'}_architecture.json`;
      a.click();
      URL.revokeObjectURL(url);
    });

    // Initialize
    loadCatalog();
  </script>
</body>
</html>
"""


# ==============================================================================
# HTTP Request Handler & Server
# ==============================================================================
class ArchitectureExplorerHandler(http.server.BaseHTTPRequestHandler):
    """Custom HTTP handler serving web interface and REST API."""

    def log_message(self, format: str, *args: Any):
        _logger.debug(f"{self.address_string()} - {format % args}")

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif path == "/api/catalog":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(CATALOG).encode("utf-8"))

        elif path == "/api/search":
            query_params = urllib.parse.parse_qs(parsed.query)
            q = query_params.get("q", [""])[0].strip()
            results = []
            if timm is not None and q:
                results = timm.list_models(f"*{q}*")[:30]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(results).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/inspect":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                data = json.loads(body.decode("utf-8"))
                model_name = data.get("model_name", "").strip()
                input_size_str = data.get("input_size")
                depth = data.get("depth", 3)
                direction = data.get("direction", "LR")
                custom_loader = data.get("custom_loader")

                if not model_name:
                    raise ValueError("Model name must not be empty.")

                _logger.info(f"Inspecting model <{model_name}> (depth={depth}, direction={direction})...")
                model = load_model(model_name, pretrained=False, custom_loader=custom_loader)
                input_size = parse_input_size(model, input_size_str)

                result = inspect_architecture(
                    model=model,
                    input_size=input_size,
                    depth=depth,
                    direction=direction,
                )

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))

            except Exception as e:
                _logger.error(f"Error inspecting model: {e}", exc_info=True)
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def find_free_port(start_port: int = 8000) -> int:
    """Find next available free port."""
    port = start_port
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
            port += 1
    return start_port


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Model Architecture Explorer with Shapes and Connections."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model architecture name to open directly (e.g. vit_base_patch16_clip_224.openai_ft_in12k_in1k)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run web server on (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open web browser",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Export inspection to JSON file directly and exit without starting server",
    )
    args = parser.parse_args()

    # Direct CLI Export mode
    if args.export_json:
        if not args.model:
            parser.error("--export-json requires --model to be specified.")
        print(f"Loading and inspecting {args.model}...")
        model = load_model(args.model, pretrained=False)
        input_size = parse_input_size(model, None)
        res = inspect_architecture(model, input_size, depth=3)
        with open(args.export_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Architecture inspection saved to {args.export_json}")
        sys.exit(0)

    # Server mode
    port = find_free_port(args.port)
    server_address = (args.host, port)
    httpd = http.server.ThreadingHTTPServer(server_address, ArchitectureExplorerHandler)

    url = f"http://{args.host}:{port}"
    print(f"\n{'='*70}")
    print(f"🚀 Model Architecture Explorer running at: {url}")
    print(f"{'='*70}\n")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    main()
