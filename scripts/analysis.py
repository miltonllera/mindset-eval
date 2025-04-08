import sys
import re
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import statsmodels.api as sm

sys.path.extend(["../", "../mindset"])

import numpy as np
import pandas as pd


#-------------------------------------- Data Loading ---------------------------------------------

def get_recording_files(results_folder: Path, model_names: str, metric='cossim'):
    return [(results_folder / n / f"{metric}_df.csv") for n in model_names]


def get_infor_from_recording_file_path(file_path: Path):
    model_name, file_name = file_path.parts[-2:]
    metric = file_name.split('_')[0]
    return model_name, metric


def load_data(recording_path, list_comparison_levels=None):
    list_comparison_levels = None

    df = pd.read_csv(recording_path)
    df = df.rename(columns={'Unnamed: 0': 'Entry'})

    pattern = re.compile(r"^\d+: .*$")
    layers_names = [col for col in df.columns if pattern.match(col)]

    if list_comparison_levels is not None:
        df = df[df["ComparisonLevel"].isin(list_comparison_levels)]

    df = df.drop(columns=["ReferenceLevel", "ReferencePath", "ComparisonPath"])

    return df, layers_names


#---------------------------------------- Plotting -----------------------------------------------

def plot_per_layer_distance(model_name, metric, df, start_from=0):
    mean_distances = df.groupby("ComparisonLevel").mean().filter(regex="(Conv|Linear)")

    mean_distances_std = df.groupby("ComparisonLevel").std().filter(regex="(Conv|Linear)")

    plt.rcParams["svg.fonttype"] = "none"
    sns.set_style("white")

    conditions = [
        np.array(mean_distances.iloc[i])[start_from:] for i in range(len(mean_distances))
    ]
    conditions_std = [
        np.array(mean_distances_std.iloc[i])[start_from:]
        for i in range(len(mean_distances))
    ]
    layer_index_name = mean_distances.columns[start_from:]

    x = range(len(conditions[0]))
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    plt.axhline(y=0, linestyle=":", color="k", alpha=0.6, linewidth=1)

    for idx, c in enumerate(conditions):
        plt.plot(x, c, label=mean_distances.index[idx], lw=2)

    ax.set_xticks([])
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=12)

    ax.patch.set_edgecolor("black")
    ax.patch.set_linewidth(1)
    sns.despine()
    plt.legend(prop={"size": 12}, edgecolor=(0, 0, 0, 1))
    plt.ylabel(metric, fontsize=12)
    plt.xlabel("Network Depth", fontsize=12)
    ax.annotate(
        model_name,
        color="k",
        xy=(0.2, 0.95),
        xycoords="axes fraction",
        size=15,
        fontweight="bold",
        bbox=dict(boxstyle="round", alpha=1, fc="wheat", ec="k"),
    )
    ax.tick_params(
        axis="y",
        left=True,
        labelbottom=True,
    )
    ax.tick_params(
        axis="x",
        bottom=True,
        labelbottom=True,
    )
    ax.set_xticks(x[::25], layer_index_name[::25], rotation=0)

    plt.tight_layout()
    return fig


def box_plot_bunch_at_position(
    sample, x, ax, ll=None, cc=None, width=None, span=None, space=None, showmeans=True
):
    span = 0.35 if span is None else span
    space = 0.03 if space is None else space
    mm = len(sample) if len(sample) > 1 else 2
    width = span / (mm - 1) if width is None else width
    i = np.arange(0, len(sample))
    ll = [None] * len(sample) if ll is None else ll
    for iidx, d in enumerate(sample):
        bp = ax.boxplot(
            d,
            patch_artist=True,
            showfliers=False,
            positions=[x - span / 2 + width * i[iidx]],
            widths=width - space,
            labels=[ll[iidx]],
            boxprops={
                "fill": (0, 0, 0, 1),
                "lw": 0.5,
                "facecolor": f"C{iidx}" if cc is None else cc,
                "alpha": 1,
            },
            showmeans=showmeans,
            meanprops={"markerfacecolor": "r", "markeredgecolor": "k"},
            medianprops={"color": "k", "linestyle": "-", "linewidth": 0.5, "zorder": 2},
        )


def plot_layer_comparison(df, lname):
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.tick_params(
        axis="y",
        left=True,
        labelbottom=True,
    )

    cl = df["ComparisonLevel"].unique()
    dd = np.array([df[(df["ComparisonLevel"] == i)][lname] for i in cl])
    [box_plot_bunch_at_position([d], 0, ax, cc="C0", width=None) for d in dd]
    ax.set_xticklabels(cl)
    plt.ylabel("Distance Metric")
    return fig


#------------------------------------------- Stats ------------------------------------------------

def compute_layer_t_test(df, subject, within, name_layer_used):
    model = sm.formula.ols(f'Q("{name_layer_used}") ~ C({within})', data=df).fit()
    print(model.summary())
    t_test_result = model.t_test(f"C({within})[T.{subject}] = 0")
    print(t_test_result)
    return t_test_result
