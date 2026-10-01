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
    logging.getLogger("kaleido").setLevel(logging.ERROR)
    logging.getLogger("choreographer").setLevel(logging.ERROR)
    return _logger


def init_model(model_name, pretrained=True, verbose=False):
    model = timm.create_model(
        model_name, pretrained=pretrained, cache_dir="data/models/"
    )

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
    return transforms.Compose(
        [
            transforms.Resize(cfg["input_size"][-1]),
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ]
    )


def get_recording_files(
    results_folder: Path, model_names: list[str] | str, metric: str
):
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
                [
                    df.filter(pl.col(condition_col) == grp)[l].mean()
                    for l in layer_names
                ],
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

    out_file = results_folder / (
        filename if filename is not None else f"{metric}_vs_layer.png"
    )
    fig.write_image(out_file)
    return out_file


def plot_uncrowding_grid_arrangements(
    df,
    results_folder: Path | str,
    layer_names: list[str] | None = None,
    metric: str = "Accuracy",
    filename: str | None = None,
    max_cols: int = 3,
) -> Path:
    results_folder = Path(results_folder)
    if hasattr(df, "compute"):
        pdf = df.compute()
    elif isinstance(df, pl.DataFrame):
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    if "NumRows" not in pdf.columns or "NumCols" not in pdf.columns:
        dims = pdf["GridPattern"].astype(str).str.split(":").str[0].str.split("x")
        pdf["NumRows"], pdf["NumCols"] = dims.str[0].astype(int), dims.str[1].astype(
            int
        )

    if layer_names is None:
        exclude = {
            "SampleID",
            "VernierOffset",
            "GridPattern",
            "GridArrangement",
            "NumRows",
            "NumCols",
            "Target",
            "Pattern Length",
            "ShapeSize",
            "Path",
            "VernierType",
            "IterNum",
            "BackgroundColor",
        }
        layer_names = [
            c for c in pdf.columns if c not in exclude and not str(c).startswith("__")
        ]

    agg = pdf.groupby(["NumRows", "NumCols", "GridArrangement"])[layer_names].agg(
        ["mean", "std"]
    )
    grid_sizes = sorted(
        set((r, c) for r, c, _ in agg.index), key=lambda x: (x[0] * x[1], x[0], x[1])
    )

    n_cols = min(max_cols, len(grid_sizes)) or 1
    n_rows = math.ceil(len(grid_sizes) / n_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"{r}x{c} Grid" for r, c in grid_sizes],
        shared_yaxes=True,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    colors = {
        "uniform": "#1f77b4",
        "interleaved_cols": "#ff7f0e",
        "interleaved_rows": "#2ca02c",
        "interleaved_both": "#d62728",
    }
    seen_legend = set()

    for idx, (r, c) in enumerate(grid_sizes):
        row, col = (idx // n_cols) + 1, (idx % n_cols) + 1
        arrangements = [
            a
            for a in [
                "uniform",
                "interleaved_cols",
                "interleaved_rows",
                "interleaved_both",
            ]
            if (r, c, a) in agg.index
        ]

        for arr in arrangements:
            means = agg.loc[
                (r, c, arr), [(l, "mean") for l in layer_names]
            ].values.astype(float)
            stds = (
                agg.loc[(r, c, arr), [(l, "std") for l in layer_names]]
                .fillna(0)
                .values.astype(float)
            )
            upper = np.clip(means + stds, 0.0, 1.0)
            lower = np.clip(means - stds, 0.0, 1.0)

            color = colors.get(arr, "#7f7f7f")
            fig.add_trace(
                go.Scatter(
                    x=layer_names + layer_names[::-1],
                    y=list(upper) + list(lower[::-1]),
                    fill="toself",
                    fillcolor=_color_to_rgba(color, 0.12),
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=arr,
                ),
                row=row,
                col=col,
            )

            fig.add_trace(
                go.Scatter(
                    x=layer_names,
                    y=means,
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=arr.replace("_", " ").title(),
                    legendgroup=arr,
                    showlegend=(arr not in seen_legend),
                ),
                row=row,
                col=col,
            )
            seen_legend.add(arr)

    fig.update_layout(
        template="plotly_white",
        height=300 * n_rows,
        width=400 * n_cols,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        margin=dict(l=60, r=40, t=80, b=90),
    )
    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        tickangle=-45,
        tickfont=dict(size=9 if len(layer_names) <= 25 else 7),
    )
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor="black", mirror=True, range=[-0.02, 1.02]
    )
    for r_i in range(1, n_rows + 1):
        fig.update_yaxes(title_text=metric, row=r_i, col=1)
    for c_i in range(1, n_cols + 1):
        fig.update_xaxes(title_text="Layer", row=n_rows, col=c_i)
    fig.for_each_annotation(lambda a: a.update(font=dict(size=13, color="black")))

    out_file = results_folder / (
        filename if filename is not None else f"{metric}_grid_arrangements.png"
    )
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
    Plots the similarity difference Sim(Ref, MP) - Sim(Ref, NAP) as a function of |RefVal|
    with one panel per Dimension and layers displayed along a dark-to-light gradient
    (where deeper layers are lighter). Y-axis scale is constant across panels, while
    each dimension has its own x-axis scale.
    """
    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    pdf = (
        df.compute()
        if hasattr(df, "compute")
        else (df.to_pandas() if isinstance(df, pl.DataFrame) else df.copy())
    )
    if layer_names is None:
        exclude = {
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
        layer_names = [c for c in pdf.columns if c not in exclude]

    # Align comparison conditions and compute difference
    mp = pdf[pdf["Comparison"].astype(str).str.contains("MP", case=False)].set_index(
        "SampleID"
    )
    nap = pdf[pdf["Comparison"].astype(str).str.contains("NAP", case=False)].set_index(
        "SampleID"
    )
    common_ids = mp.index.intersection(nap.index)
    diff = (
        mp.loc[common_ids, layer_names].astype(float)
        - nap.loc[common_ids, layer_names].astype(float)
    ).assign(
        RefVal=pd.to_numeric(mp.loc[common_ids, "RefVal"], errors="coerce").abs(),
        Dimension=(
            mp.loc[common_ids, "Dimension"].astype(str)
            if "Dimension" in mp.columns
            else "Default"
        ),
    )

    unique_dims, n_layers = list(diff["Dimension"].unique()), len(layer_names)
    start_rgb, end_rgb = (8, 48, 107), (142, 199, 232)
    colors = [
        f"rgb({int(start_rgb[0] + t * (end_rgb[0] - start_rgb[0]))},"
        f"{int(start_rgb[1] + t * (end_rgb[1] - start_rgb[1]))},"
        f"{int(start_rgb[2] + t * (end_rgb[2] - start_rgb[2]))})"
        for t in [i / max(1, n_layers - 1) for i in range(n_layers)]
    ]
    tick_idx = (
        list(range(n_layers))
        if n_layers <= 8
        else sorted(set(np.linspace(0, n_layers - 1, 7, dtype=int)))
    )

    # Global y-range across all layers for constant comparison
    y_min, y_max = min(0.0, float(diff[layer_names].min().min())), max(
        0.0, float(diff[layer_names].max().max())
    )
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.1

    ncols = min(max_cols, len(unique_dims))
    nrows = math.ceil(len(unique_dims) / ncols)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[str(d) for d in unique_dims],
        horizontal_spacing=max(0.04, 0.25 / ncols),
        vertical_spacing=max(0.06, 0.4 / nrows),
    )

    # Colorbar on side
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            showlegend=False,
            hoverinfo="none",
            marker=dict(
                colorscale=[[0.0, f"rgb{start_rgb}"], [1.0, f"rgb{end_rgb}"]],
                cmin=0,
                cmax=max(1, n_layers - 1),
                color=[0, max(1, n_layers - 1)],
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Layer Depth<br>(Early → Deep)", font=dict(size=11)
                    ),
                    tickmode="array",
                    tickvals=tick_idx,
                    ticktext=[layer_names[i] for i in tick_idx],
                    tickfont=dict(size=9),
                    len=0.85,
                    y=0.5,
                    yanchor="middle",
                ),
            ),
        ),
        row=1,
        col=1,
    )

    for i, dim in enumerate(unique_dims):
        r, c = (i // ncols) + 1, (i % ncols) + 1
        dim_df = diff[diff["Dimension"] == dim]
        agg = dim_df.groupby("RefVal")[layer_names].agg(["mean", "std"]).sort_index()
        xs = agg.index.tolist()
        if not xs:
            continue

        x_pad = 0.05 * (max(xs) - min(xs)) if max(xs) > min(xs) else 0.5
        x_range = [max(0.0, min(xs) - x_pad), max(xs) + x_pad]

        # Zero reference line
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=[0, 0],
                mode="lines",
                line=dict(color="rgba(128,128,128,0.6)", width=1, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=r,
            col=c,
        )

        for l_idx, layer in enumerate(layer_names):
            means, stds = agg[(layer, "mean")].tolist(), agg[(layer, "std")].tolist()
            color = colors[l_idx]
            if any(pd.notna(s) and s > 0 for s in stds):
                upper = [(m + s) if pd.notna(s) else m for m, s in zip(means, stds)]
                lower = [(m - s) if pd.notna(s) else m for m, s in zip(means, stds)]
                fig.add_trace(
                    go.Scatter(
                        x=xs + xs[::-1],
                        y=upper + lower[::-1],
                        fill="toself",
                        fillcolor=_color_to_rgba(color, 0.15),
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=r,
                    col=c,
                )
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=means,
                    mode="lines+markers",
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    name=str(layer),
                    showlegend=False,
                    hovertemplate=f"Dimension: {dim}<br>Layer: {layer} (depth {l_idx})<br>|RefVal|: %{{x}}<br>ΔSim: %{{y:.4f}}<extra></extra>",
                ),
                row=r,
                col=c,
            )

        fig.update_xaxes(
            range=x_range,
            title_text="|RefVal|",
            title_font=dict(size=10),
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            tickfont=dict(size=9),
            row=r,
            col=c,
        )
        if c == 1:
            fig.update_yaxes(
                title_text="ΔSim (MP - NAP)", title_font=dict(size=10), row=r, col=c
            )

    fig.update_yaxes(
        range=[y_min - y_pad, y_max + y_pad],
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        tickfont=dict(size=9),
    )
    fig.update_layout(
        template="plotly_white",
        width=max(800, ncols * 280 + 120),
        height=max(360, nrows * 280),
        title=dict(
            text=f"Similarity Difference [Sim(Ref, MP) - Sim(Ref, NAP)] vs |RefVal| by Dimension ({metric})",
            x=0.5,
            xanchor="center",
            font=dict(size=14),
        ),
        margin=dict(l=60, r=120, t=70, b=50),
    )
    fig.for_each_annotation(lambda a: a.update(font=dict(size=11)))

    out_file = results_folder / (
        filename if filename is not None else f"{metric}_diff_vs_refval.png"
    )
    fig.write_image(out_file)
    return out_file
