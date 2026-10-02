import argparse
import shutil
from math import prod
from pathlib import Path

import torch
import polars as pl
import dask.dataframe as dd
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.dataset import AnnotatedDataset
from src.feature_decoder import FeatureDecoder
from src.utils import (
    get_device,
    model_transform,
    init_model,
    get_recording_files,
    plot_layer_scores,
    plot_uncrowding_grid_arrangements,
    setup_logging
)

torch.set_float32_matmul_precision('high')

_logger = setup_logging(__name__)

TEST_COLUMNS = [
    'Path', 'VernierType', 'VernierOffset', 'GridPattern', 'NumRows', 'NumCols',
    'CenterShape', 'AlternateShape', 'Loc', 'Scale'
]
TARGET_COLUMN = 'VernierType'


def record_from_model(
    recorder: tuple[str, torch.nn.Module],
    dataloader: DataLoader,
    results_folder: Path,
    output_tag: str | None = None,
    flush_every_n_batches: int = 10,
):
    model_name, feature_extractor = recorder
    results_folder = results_folder / model_name
    results_folder.mkdir(parents=True, exist_ok=True)

    _logger.info(f"Recording from model: <{model_name}>")

    device = get_device()

    tag_suffix = f"_{output_tag}" if output_tag else ""
    recordings_file_path = results_folder / f"predictions{tag_suffix}.parquet"
    if recordings_file_path.exists():
        if recordings_file_path.is_dir():
            shutil.rmtree(recordings_file_path)
        else:
            recordings_file_path.unlink()

    buffer_chunks = []

    def _flush_buffer():
        nonlocal buffer_chunks
        if buffer_chunks:
            flush_df = pl.concat(buffer_chunks)
            dd.from_pandas(flush_df, npartitions=1).to_parquet(
                recordings_file_path,
                engine="pyarrow",
                append=True,
                ignore_divisions=True,
            )
            buffer_chunks.clear()
            del flush_df

    feature_extractor = feature_extractor.eval().to(device=device)
    sample_counter = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=model_name)):
            images = batch['Image'].to(device)
            preds = feature_extractor(images)
            targets = batch[TARGET_COLUMN]
            bsz = len(images)

            sample_ids = list(range(sample_counter, sample_counter + bsz))
            sample_counter += bsz

            chunk_dict = {k: list(batch[k]) for k in TEST_COLUMNS if k in batch}
            chunk_dict['SampleID'] = sample_ids
            chunk_dict['Target'] = [int(t) if isinstance(t, torch.Tensor) else t for t in targets]

            for k, v in preds.items():
                pred_tensor = v.squeeze(-1) if v.ndim > 1 else v
                chunk_dict[k] = (pred_tensor.sigmoid() > 0.5).to(dtype=torch.int).cpu().tolist()

            buffer_chunks.append(pl.DataFrame(chunk_dict))

            del images, preds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (batch_idx + 1) % flush_every_n_batches == 0:
                _flush_buffer()

        _flush_buffer()

    # Consolidate partitions to eliminate fragmentation
    ddf = dd.read_parquet(recordings_file_path)
    if ddf.npartitions > 1:
        temp_consolidated_path = results_folder / f"predictions{tag_suffix}_consolidated.parquet"
        ddf.repartition(npartitions=1).to_parquet(
            temp_consolidated_path,
            engine="pyarrow",
        )
        shutil.rmtree(recordings_file_path)
        temp_consolidated_path.rename(recordings_file_path)
        ddf = dd.read_parquet(recordings_file_path)

    layer_names = [
        c for c in ddf.columns if c not in TEST_COLUMNS and not str(c).startswith('__')
    ]
    for layer in layer_names:
        ddf[layer] = (ddf[layer] == ddf['Target']).astype(float)

    ddf['Pattern Length'] = ddf['GridPattern'].map(
        lambda x: prod(map(int, x.split(':')[0].split('x'))),
        meta=('GridPattern', 'int64')
    )

    plot_filename = f"accuracy_vs_layer{tag_suffix}.png"
    if 'GridArrangement' in ddf.columns:
        plot_uncrowding_grid_arrangements(
            ddf,
            results_folder=results_folder,
            layer_names=layer_names,
            metric="Accuracy",
            filename=plot_filename,
        )
    else:
        plot_layer_scores(
            ddf,
            metric="Accuracy",
            results_folder=results_folder,
            layer_names=layer_names,
            condition_col="Pattern Length",
            filename=plot_filename,
        )

    _logger.info(f"Recording finished. Saved to: <{recordings_file_path}>")
    return recordings_file_path


def train_feature_extractor(feature_decoder, dataset):
    train_data, val_data = random_split(dataset, [0.8, 0.2])
    dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=0)
    trainer = Trainer(
        max_epochs=-1, max_steps=1000, val_check_interval=100, check_val_every_n_epoch=None
    )
    trainer.fit(feature_decoder, train_dataloaders=dataloader, val_dataloaders=val_dataloader)
    return feature_decoder


def record_all(annotations_file, model_names, record_from, results_folder, output_tag=None):
    recording_paths = []
    for model_name in model_names:
        model = init_model(model_name)
        feature_decoder = FeatureDecoder(
            model, target_dim=1, target_key='VernierType', loss='cross_entropy',
            decode_from=record_from,
        )

        if not feature_decoder._hooks:
            _logger.warning(
                f"No layers matched record_from pattern {record_from} for <{model_name}>. Skipping."
            )
            continue

        available_cols = pl.read_csv(annotations_file, n_rows=1).columns
        test_columns = [ c for c in [
                'Path', 'VernierType', 'VernierOffset', 'GridPattern', 'GridArrangement',
                'NumRows', 'NumCols', 'ShapeSize']
            if c in available_cols
        ]

        train_dataset = AnnotatedDataset(
            annotations_file,
            test_columns=test_columns,
            transform=model_transform(model),
            filter_expr="pl.col('VernierInOut') == 'outside'",
        )
        feature_decoder = train_feature_extractor(feature_decoder, train_dataset)

        test_dataset = AnnotatedDataset(
            annotations_file,
            test_columns=test_columns,
            transform=model_transform(model),
            filter_expr="pl.col('VernierInOut') == 'inside'",
        )
        dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        recording_paths.append(
            record_from_model(
                (model_name, feature_decoder),
                dataloader,
                results_folder,
                output_tag=output_tag,
            )
        )
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

    results_folder = Path(results_folder) / 'uncrowding'

    if not results_folder.exists() or overwrite_recordings:
        results_folder.mkdir(parents=True, exist_ok=True)
        _logger.info(f"Set results root folder to {results_folder}")
        record_all(
            annotations_file,
            model_names,
            record_from,
            results_folder,
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
    parser.add_argument("--record_from", type=str, nargs='*', default=None,
        help="Regex patterns or names of layers to record from"
    )
    parser.add_argument("--output_tag", type=str, default=None,
        help="Optional tag to append to the output files (e.g. chunk_0)"
    )
    parser.add_argument("--results_folder", type=str, default='data/results',
        help="Experiment folder where to store all results"
    )
    parser.add_argument("--overwrite_recordings", action='store_true',
        help="Overwrite the recording file if it already exists"
    )
    args = parser.parse_args()
    main(**vars(args))
