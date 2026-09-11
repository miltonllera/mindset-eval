import argparse
import shutil
from pathlib import Path

import torch
import pandas as pd
import dask.dataframe as dd
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import AnnotatedDataset
from src.activation_recorder import ActivationRecorder
from src.utils import (
    get_device,
    model_transform,
    init_model,
    get_recording_files,
    plot_layer_scores,
    setup_logging
)

_logger = setup_logging(__name__)


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
    record_from: list[str] | None = None,
    output_tag: str | None = None,
    flush_every_n_batches: int = 10,
):
    model_name, net = model
    results_folder = results_folder / model_name
    results_folder.mkdir(parents=True, exist_ok=True)

    _logger.info(f"Recording from model: <{model_name}>")

    device = get_device()

    if metric == "cossim":
        calc_dist = lambda a, b, bsz: torch.nn.functional.cosine_similarity(
            a.reshape(bsz, -1), b.reshape(bsz, -1), dim=1
        ).tolist()
    elif metric == "euclidean":
        calc_dist = lambda a, b, bsz: torch.norm(
            a.reshape(bsz, -1) - b.reshape(bsz, -1), dim=1
        ).tolist()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    tag_suffix = f"_{output_tag}" if output_tag else ""
    recordings_file_path = results_folder / f"{metric}{tag_suffix}.parquet"
    if recordings_file_path.exists():
        if recordings_file_path.is_dir():
            shutil.rmtree(recordings_file_path)
        else:
            recordings_file_path.unlink()

    recorder = ActivationRecorder(net, record_from=record_from)
    if not recorder._hooks:
        _logger.warning(
            f"No layers matched record_from pattern {record_from} for <{model_name}>. Skipping."
        )
        return None

    layer_names = []
    buffer_chunks = []

    def _flush_buffer():
        nonlocal buffer_chunks
        if buffer_chunks:
            flush_df = pd.concat(buffer_chunks, ignore_index=True)
            dd.from_pandas(flush_df, npartitions=1).to_parquet(
                recordings_file_path,
                engine="pyarrow",
                append=True,
                ignore_divisions=True,
            )
            buffer_chunks.clear()
            del flush_df

    net.eval()
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=model_name)):
            bsz = len(batch['SampleID'])
            batch_layer_acts = {}
            for img_type in IMAGE_TYPES:
                images = batch[f'{img_type}Path'].to(device)
                net(images)
                batch_layer_acts[img_type] = {k: v.cpu() for k, v in recorder.activation.items()}

            layer_names = list(recorder.activation.keys())
            sample_ids = [int(x) for x in batch['SampleID']]
            shape_types = list(batch['Shape'])
            dimensions = list(batch['Dimension'])

            for ref_type, comp_type in COMPARISONS:
                ref_acts = batch_layer_acts[ref_type]
                comp_acts = batch_layer_acts[comp_type]
                chunk_dict = {
                    'SampleID': sample_ids,
                    'Shape': shape_types,
                    'Dimension': dimensions,
                    'Comparison': [f'{ref_type}_vs_{comp_type}'] * bsz,
                }
                for k in layer_names:
                    chunk_dict[k] = calc_dist(ref_acts[k], comp_acts[k], bsz)
                buffer_chunks.append(pd.DataFrame(chunk_dict))

            del batch_layer_acts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (batch_idx + 1) % flush_every_n_batches == 0:
                _flush_buffer()

        _flush_buffer()

    recorder.remove_hooks()

    # Consolidate partitions to eliminate fragmentation
    ddf = dd.read_parquet(recordings_file_path)
    if ddf.npartitions > 1:
        temp_consolidated_path = results_folder / f"{metric}{tag_suffix}_consolidated.parquet"
        ddf.repartition(npartitions=1).to_parquet(
            temp_consolidated_path,
            engine="pyarrow",
        )
        shutil.rmtree(recordings_file_path)
        temp_consolidated_path.rename(recordings_file_path)
        ddf = dd.read_parquet(recordings_file_path)

    plot_filename = f"{metric}_vs_layer{tag_suffix}.png"
    plot_layer_scores(ddf, metric, results_folder, layer_names=layer_names, filename=plot_filename)

    _logger.info(f"Recording finished. Saved to: <{recordings_file_path}>")
    return recordings_file_path


def record_all(
    annotations_file,
    model_names,
    results_folder,
    record_from=None,
    output_tag=None,
):
    recording_paths = []
    for model_name in model_names:
        model = init_model(model_name)
        dataset = AnnotatedDataset(
            annotations_file,
            test_columns=TEST_COLUMNS,
            transform=model_transform(model),
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        path = record_from_model(
            (model_name, model),
            "cossim",
            dataloader,
            results_folder,
            record_from=record_from,
            output_tag=output_tag,
        )
        if path:
            recording_paths.append(path)
    return recording_paths


def main(
    annotations_file,
    model_names,
    record_from=None,
    output_tag=None,
    results_folder='',
    overwrite_recordings=False,
):
    _logger.info("Loading models...")

    results_folder = Path(results_folder) / 'depth_drawings'

    if not results_folder.exists() or overwrite_recordings:
        results_folder.mkdir(parents=True, exist_ok=True)
        _logger.info(f"Set results root folder to {results_folder}")
        record_all(
            annotations_file,
            model_names,
            results_folder,
            record_from=record_from,
            output_tag=output_tag,
        )
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
    parser.add_argument("--record_from", type=str, nargs='+', default=None,
        help="Layer names, regex patterns, or stream specs to record from (e.g. 'stages\\.\\d+\\.blocks\\.\\d+:all')"
    )
    parser.add_argument("--output_tag", type=str, default=None,
        help="Optional tag to append to output recording and plot filenames"
    )
    parser.add_argument("--overwrite_recordings", action='store_true',
        help="Overwrite the recording file if it all already exists"
    )

    args = parser.parse_args()
    main(**vars(args))
