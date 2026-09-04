import logging
from pathlib import Path

import torch
import torchvision.transforms as transforms
import timm
import polars as pl
import plotly.express as px
import plotly.graph_objects as go


def setup_logging(module_name):
    logging.basicConfig(level=logging.INFO)
    _logger = logging.getLogger(module_name)
    logging.getLogger('kaleido').setLevel(logging.ERROR)
    logging.getLogger('choreographer').setLevel(logging.ERROR)
    return _logger


def init_model(model_name, verbose=False):
    model = timm.create_model(  # type: ignore
        model_name, pretrained=True, cache_dir="data/models/"
    ).to(get_device())

    if verbose:
        print(model)

    return model


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
        csv_path = results_folder / n / f"{metric}.csv"
        csv_df_path = results_folder / n / f"{metric}_df.csv"
        if parquet_path.exists():
            files.append(parquet_path)
        elif csv_path.exists():
            files.append(csv_path)
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
    group_col: str = "Comparison",
    filename: str | None = None,
) -> Path:
    results_folder = Path(results_folder)
    if layer_names is None:
        exclude_cols = {
            group_col,
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
        agg_df = df.groupby(group_col)[layer_names].agg(["mean", "std"]).compute()
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
        groups = df[group_col].unique(maintain_order=True).to_list()
        group_stats = [
            (
                grp,
                [df.filter(pl.col(group_col) == grp)[l].mean() for l in layer_names],
                [df.filter(pl.col(group_col) == grp)[l].std() for l in layer_names],
            )
            for grp in groups
        ]
    else:
        groups = list(df[group_col].unique())
        group_stats = [
            (
                grp,
                [df[df[group_col] == grp][l].mean() for l in layer_names],
                [df[df[group_col] == grp][l].std() for l in layer_names],
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
