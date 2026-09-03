import argparse
import logging
from pathlib import Path

import torch
import polars as pl
import plotly.express as px
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import AnnotatedDataset
from src.activation_recorder import ActivationRecorder
from src.utils import get_device, model_transform, init_model, get_recording_files, plot_layer_scores


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


TEST_COLUMNS = [
    'SampleID', 'Shape','BasisPath', 'BasisRotatedPath', 'RectangleHullPath',
    'RectangleHullRotatedPath', '3DShapePath', '3DShapeRotatedPath'
]
IMAGE_TYPES  = [
    'Basis', 'BasisRotated', 'RectangleHull', 'RectangleHullRotated', '3DShape', '3DShapeRotated'
]
COMPARISONS  = [
    ('Basis', 'BasisRotated'),
    ('RectangleHull', 'RectangleHullRotated'),
    ('3DShape', '3DShapeRotated'),
]


def record_from_model(
    model: tuple[str, torch.nn.Module],
    metric: str,
    dataloader: DataLoader,
    results_folder: Path,
):
    model_name, net = model
    results_folder = results_folder / model_name
    results_folder.mkdir(parents=True, exist_ok=True)

    _logger.info(f"Recording from model: <{model_name}>")

    device = get_device()

    if metric == "cossim":
        distance_fn = lambda a, b: torch.nn.functional.cosine_similarity(
            a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
        ).item()
    elif metric == "euclidean":
        distance_fn = lambda a, b: torch.norm(a.flatten() - b.flatten()).item()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    recorder = ActivationRecorder(net)
    all_records = []

    net.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=model_name):
            for img_type in IMAGE_TYPES:
                images = batch[f'{img_type}Image'].to(device)
                net(images)
                layer_acts = {k: v.cpu() for k, v in recorder.activation.items()}
                for i in range(len(images)):
                    all_records.append({
                        'SampleID':  int(batch['SampleID'][i]),
                        'Shape':     batch['Shape'][i],
                        'ImageType': img_type,
                        **{k: v[i] for k, v in layer_acts.items()},
                    })

    recorder.remove_hooks()

    layer_names = list(recorder.activation.keys())
    records_by_sample = {}
    for r in all_records:
        records_by_sample.setdefault(r['SampleID'], {})[r['ImageType']] = r

    df_rows = []
    for sample_id, by_type in records_by_sample.items():
        for ref_type, comp_type in COMPARISONS:
            ref, comp = by_type[ref_type], by_type[comp_type]
            df_rows.append({
                'SampleID':   sample_id,
                'Shape':      ref['Shape'],
                'Comparison': f'{ref_type}_vs_{comp_type}',
                **{k: distance_fn(ref[k], comp[k]) for k in layer_names},
            })

    df = pl.DataFrame(df_rows)
    recordings_file_path = results_folder / f"{metric}.csv"
    df.write_csv(recordings_file_path)

    plot_layer_scores(df, metric, results_folder, layer_names=layer_names)

    _logger.info(f"Recording finished. Saved to: <{recordings_file_path}>")
    return recordings_file_path


def record_all(annotations_file, model_names, results_folder):
    recording_paths = []
    for model_name in model_names:
        model = init_model(model_name)
        dataset = AnnotatedDataset(
            annotations_file,
            test_columns=TEST_COLUMNS,
            transform=model_transform(model),
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        recording_paths.append(
            record_from_model((model_name, model), "cossim", dataloader, results_folder)
        )
    return recording_paths


def main(
    annotations_file,
    model_names,
    results_folder='',
    overwrite_recordings=False,
):
    _logger.info("Loading models...")

    results_folder = Path(results_folder) / 'depth_drawings'

    if not results_folder.exists() or overwrite_recordings:
        results_folder.mkdir(parents=True, exist_ok=True)
        _logger.info(f"Set results root folder to {results_folder}")
        record_all(annotations_file, model_names, results_folder)
    else:
        get_recording_files(results_folder, model_names, 'cossim')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", type=str, nargs='+', dest='model_names',
        help="List of models to test"
    )
    parser.add_argument("--annotations_file", type=str,
        help="Path to the annotations file used to run the experiment."
    )
    parser.add_argument("--results_folder", type=str, default='data/results',
        help="Experiment folder where to store all results"
    )
    parser.add_argument("--overwrite_recordings", action='store_true',
        help="Overwrite the recording file if it all already exists"
    )

    args = parser.parse_args()
    main(**vars(args))
