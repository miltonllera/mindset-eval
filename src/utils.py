import logging
from pathlib import Path

import math
import torch
import torchvision.transforms as transforms
import timm
import numpy as np
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def setup_logging(module_name):
    logging.basicConfig(level=logging.INFO)
    _logger = logging.getLogger(module_name)
    logging.getLogger('kaleido').setLevel(logging.ERROR)
    logging.getLogger('choreographer').setLevel(logging.ERROR)
    return _logger


def init_model(model_name, pretrained=True, verbose=False):
    model = timm.create_model(model_name, pretrained=pretrained, cache_dir="data/models/")

    # Disable inplace modification so recording for pre-activations returns the correct tensors.
    for m in model.modules():
        if isinstance(m, torch.nn.ReLU):
            m.inplace = False

    if verbose:
        print(model)

    return model.to(get_device())


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def model_transform(model) -> transforms.Compose:
    cfg = model.pretrained_cfg
    return transforms.Compose([
        transforms.Resize(cfg["input_size"][-1]),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])


def get_recording_files(results_folder: Path, model_names: list[str] | str, metric: str):
    if isinstance(model_names, str):
        model_names = [model_names]
    files = []
    for n in model_names:
        parquet_path = results_folder / n / f"{metric}.parquet"
        pred_parquet_path = results_folder / n / "predictions.parquet"
        csv_path = results_folder / n / f"{metric}.csv"
        pred_csv_path = results_folder / n / "predictions.csv"
        csv_df_path = results_folder / n / f"{metric}_df.csv"
        if parquet_path.exists():
            files.append(parquet_path)
        elif pred_parquet_path.exists():
            files.append(pred_parquet_path)
        elif csv_path.exists():
            files.append(csv_path)

        elif pred_csv_path.exists():
            files.append(pred_csv_path)
        else:
            files.append(csv_df_path)
    return files


def _color_to_rgba(color_str: str, alpha: float = 0.2) -> str:
    if color_str.startswith("#"):
        hex_str = color_str.lstrip("#")
        r, g, b = tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"
    elif color_str.startswith("rgb("):
        return color_str.replace("rgb(", "rgba(").replace(")", f", {alpha})")
    return f"rgba(100, 100, 100, {alpha})"


def plot_layer_scores(
    df,
    metric: str,
    results_folder: Path | str,
    layer_names: list[str] | None = None,
    condition_col: str = "Comparison",
    filename: str | None = None,
) -> Path:
    results_folder = Path(results_folder)
    if layer_names is None:
        exclude_cols = {
            condition_col,
            "SampleID",
            "ManipulatedShape",
            "ShapeType",
            "Shape",
            "Dimension",
            "Target",
        }
        layer_names = [c for c in df.columns if c not in exclude_cols]

    colors = px.colors.qualitative.Plotly

    # Support Dask DataFrame (lazy out-of-core groupby), Polars, or Pandas
    if hasattr(df, "compute"):
        agg_df = df.groupby(condition_col)[layer_names].agg(["mean", "std"]).compute()
        groups = list(agg_df.index)
        group_stats = [
            (
                grp,
                [agg_df.loc[grp, (l, "mean")] for l in layer_names],
                [agg_df.loc[grp, (l, "std")] for l in layer_names],
            )
            for grp in groups
        ]
    elif isinstance(df, pl.DataFrame):
        groups = df[condition_col].unique(maintain_order=True).to_list()
        group_stats = [
            (
                grp,
                [df.filter(pl.col(condition_col) == grp)[l].mean() for l in layer_names],
                [df.filter(pl.col(condition_col) == grp)[l].std() for l in layer_names],
            )
            for grp in groups
        ]
    else:
        groups = list(df[condition_col].unique())
        group_stats = [
            (
                grp,
                [df[df[condition_col] == grp][l].mean() for l in layer_names],
                [df[df[condition_col] == grp][l].std() for l in layer_names],
            )
            for grp in groups
        ]

    fig = go.Figure()

    for i, (grp, means, stds) in enumerate(group_stats):

        upper = [
            (m + s) if (m is not None and s is not None) else m
            for m, s in zip(means, stds)
        ]
        lower = [
            (m - s) if (m is not None and s is not None) else m
            for m, s in zip(means, stds)
        ]

        color = colors[i % len(colors)]
        fillcolor = _color_to_rgba(color, 0.2)

        # Continuous error band (mean +/- std)
        fig.add_trace(
            go.Scatter(
                x=layer_names + layer_names[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor=fillcolor,
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name=str(grp),
            )
        )

        # Mean line
        fig.add_trace(
            go.Scatter(
                x=layer_names,
                y=means,
                mode="lines",
                line=dict(color=color, width=2),
                name=str(grp),
            )
        )

    fig.update_layout(
        xaxis_title="Layer",
        yaxis_title=metric,
        template="plotly_white",
    )

    fig.update_xaxes(
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,  # Encloses top border
        )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,  # Encloses right border
    )

    out_file = results_folder / (filename if filename is not None else f"{metric}_vs_layer.png")
    fig.write_image(out_file)
    return out_file


def plot_refval_vs_similarity_diff(
    df,
    metric: str,
    results_folder: Path | str,
    layer_names: list[str] | None = None,
    filename: str | None = None,
    max_cols: int = 8,
) -> Path:
    """
    Plots the similarity difference Sim(Ref, MP) - Sim(Ref, NAP) as a function of RefVal
    for each layer across a grid of subplots (up to max_cols panels per row),
    grouped by Dimension with distinct colors, and with consistent x- and y-axis scales.

    Parameters:
        df: DataFrame (Dask, Polars, or Pandas) containing evaluation records.
        metric: Metric name (e.g. 'cossim').
        results_folder: Destination folder for the output plot.
        layer_names: Optional list of layer names to plot. If None, inferred automatically.
        filename: Optional filename for the output plot.
        max_cols: Maximum number of panels per row (defaults to 8).
    """
    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    if hasattr(df, "compute"):
        pdf = df.compute()
    elif isinstance(df, pl.DataFrame):
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    if layer_names is None:
        exclude_cols = {
            "Comparison",
            "SampleID",
            "ManipulatedShape",
            "ShapeType",
            "Shape",
            "Dimension",
            "RefVal",
            "MPVal",
            "NAPVal",
            "ReferencePath",
            "MPPath",
            "NAPPath",
            "Target",
        }
        layer_names = [c for c in pdf.columns if c not in exclude_cols]

    if not layer_names:
        raise ValueError("No layer names found to plot.")

    # Identify MP and NAP rows
    mp_mask = pdf["Comparison"].astype(str).str.endswith("_vs_MP") | (
        pdf["Comparison"] == "Reference_vs_MP"
    )
    nap_mask = pdf["Comparison"].astype(str).str.endswith("_vs_NAP") | (
        pdf["Comparison"] == "Reference_vs_NAP"
    )

    if not mp_mask.any() or not nap_mask.any():
        mp_mask = pdf["Comparison"].astype(str).str.contains("MP", case=False)
        nap_mask = pdf["Comparison"].astype(str).str.contains("NAP", case=False)

    mp_rows = pdf[mp_mask]
    nap_rows = pdf[nap_mask]

    if mp_rows.empty or nap_rows.empty:
        raise ValueError("Could not find both MP and NAP comparison conditions in data.")

    if "SampleID" in pdf.columns:
        mp_aligned = mp_rows.set_index("SampleID")
        nap_aligned = nap_rows.set_index("SampleID")
        common_ids = mp_aligned.index.intersection(nap_aligned.index)
        mp_aligned = mp_aligned.loc[common_ids]
        nap_aligned = nap_aligned.loc[common_ids]
    else:
        mp_aligned = mp_rows.reset_index(drop=True)
        nap_aligned = nap_rows.reset_index(drop=True)

    # Compute difference: Sim(Ref, MP) - Sim(Ref, NAP)
    diff_df = pd.DataFrame(index=mp_aligned.index)
    diff_df["RefVal"] = pd.to_numeric(mp_aligned["RefVal"], errors="coerce")
    if "Dimension" in mp_aligned.columns:
        diff_df["Dimension"] = mp_aligned["Dimension"].astype(str).values
    else:
        diff_df["Dimension"] = "Default"

    for l in layer_names:
        diff_df[l] = (
            pd.to_numeric(mp_aligned[l], errors="coerce")
            - pd.to_numeric(nap_aligned[l], errors="coerce")
        )

    # Dimensions and color mapping
    unique_dims = list(diff_df["Dimension"].unique())
    colors = px.colors.qualitative.Plotly
    dim_colors = {
        dim: colors[idx % len(colors)] for idx, dim in enumerate(unique_dims)
    }

    # Global axis limits across all panels for constant scale
    valid_refvals = diff_df["RefVal"].dropna()
    if not valid_refvals.empty:
        global_x_min = valid_refvals.min()
        global_x_max = valid_refvals.max()
        x_span = global_x_max - global_x_min
        x_pad = 0.05 * x_span if x_span > 0 else 0.5
        x_range = [global_x_min - x_pad, global_x_max + x_pad]
    else:
        x_range = None

    all_y_min = 0.0
    all_y_max = 0.0
    has_layer_data = False
    for l in layer_names:
        col_vals = diff_df[l].dropna()
        if not col_vals.empty:
            has_layer_data = True
            all_y_min = min(all_y_min, float(col_vals.min()))
            all_y_max = max(all_y_max, float(col_vals.max()))

    if has_layer_data:
        y_span = all_y_max - all_y_min
        y_pad = 0.05 * y_span if y_span > 0 else 0.1
        y_range = [all_y_min - y_pad, all_y_max + y_pad]
    else:
        y_range = None

    n_layers = len(layer_names)
    ncols = min(max_cols, n_layers)
    nrows = math.ceil(n_layers / ncols)

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=layer_names,
        horizontal_spacing=max(0.02, 0.2 / ncols),
        vertical_spacing=max(0.04, 0.4 / nrows),
    )

    for i, layer in enumerate(layer_names):
        row = (i // ncols) + 1
        col = (i % ncols) + 1

        # Zero reference line across panel width
        line_x = x_range if x_range is not None else [0, 1]
        fig.add_trace(
            go.Scatter(
                x=line_x,
                y=[0, 0],
                mode="lines",
                line=dict(color="rgba(128, 128, 128, 0.6)", width=1, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        for dim in unique_dims:
            dim_data = diff_df[diff_df["Dimension"] == dim]
            agg = (
                dim_data.groupby("RefVal")[layer]
                .agg(["mean", "std", "count"])
                .reset_index()
                .sort_values("RefVal")
            )
            ref_vals = agg["RefVal"].tolist()
            means = agg["mean"].tolist()
            stds = agg["std"].tolist()

            if not ref_vals:
                continue

            color = dim_colors[dim]
            fillcolor = _color_to_rgba(color, 0.2)

            # Shaded error band if multiple samples exist per RefVal
            has_valid_std = any(pd.notna(s) and s > 0 for s in stds)
            if has_valid_std:
                upper = [(m + s) if pd.notna(s) else m for m, s in zip(means, stds)]
                lower = [(m - s) if pd.notna(s) else m for m, s in zip(means, stds)]
                fig.add_trace(
                    go.Scatter(
                        x=ref_vals + ref_vals[::-1],
                        y=upper + lower[::-1],
                        fill="toself",
                        fillcolor=fillcolor,
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=str(dim),
                    ),
                    row=row,
                    col=col,
                )

            # Mean curve with markers
            fig.add_trace(
                go.Scatter(
                    x=ref_vals,
                    y=means,
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    name=str(dim),
                    showlegend=(i == 0),
                    legendgroup=str(dim),
                    hovertemplate=(
                        f"Layer: {layer}<br>"
                        f"Dimension: {dim}<br>"
                        "RefVal: %{x}<br>"
                        "ΔSim: %{y:.4f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        # Axis styling for individual panels
        is_bottom = (row == nrows) or ((i + ncols) >= n_layers)
        if is_bottom:
            fig.update_xaxes(title_text="RefVal", title_font=dict(size=10), row=row, col=col)
        if col == 1:
            fig.update_yaxes(
                title_text="ΔSim (MP - NAP)", title_font=dict(size=10), row=row, col=col
            )

    fig.update_layout(
        template="plotly_white",
        width=max(1200, ncols * 220),
        height=max(340, nrows * 250),
        title=dict(
            text=f"Similarity Difference [Sim(Ref, MP) - Sim(Ref, NAP)] vs RefVal ({metric})",
            x=0.5,
            xanchor="center",
            font=dict(size=14),
        ),
        legend=dict(
            title_text="Dimension",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=40, t=90, b=50),
    )

    fig.for_each_annotation(lambda a: a.update(font=dict(size=10)))

    # Set consistent axis scales across all subplots
    if x_range is not None:
        fig.update_xaxes(range=x_range)
    if y_range is not None:
        fig.update_yaxes(range=y_range)

    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        tickfont=dict(size=9),
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        tickfont=dict(size=9),
    )

    out_file = results_folder / (
        filename if filename is not None else f"{metric}_diff_vs_refval.png"
    )
    fig.write_image(out_file)
    return out_file

