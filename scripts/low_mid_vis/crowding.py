import argparse
import logging
from pathlib import Path

import torch
import polars as pl
import plotly.express as px
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import AnnotatedDataset
from src.feature_decoder import FeatureDecoder
from src.utils import get_device, model_transform, init_model, get_recording_files


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

TEST_COLUMNS = ['Path', 'VernierType', 'VernierOffset', 'GridPattern', 'ShapeSize']
TARGET_COLUMN = 'VernierType'


def record_from_model(
    recorder: tuple[str, torch.nn.Module],
    dataloader: DataLoader,
    results_folder: Path,
):
    model_name, feature_extractor = recorder
    results_folder = results_folder / model_name
    results_folder.mkdir(parents=True, exist_ok=True)

    _logger.info(f"Recording from model: <{model_name}>")

    device = get_device()

    all_records = []

    feature_extractor.eval()
    with torch.no_grad():
        n = 0
        for batch in tqdm(dataloader, desc=model_name):
            images = batch['Image'].to(device)
            preds = feature_extractor(images)
            targets = batch[TARGET_COLUMN]

            for i in range(len(images)):
                all_records.append({
                    'SampleID': (n := n + i),
                    'VernierOffset': batch['VernierOffset'][i],
                    'GridPattern': batch['GridPattern'][i],
                    'Target':  targets[i],
                    **{k: v[i] for k, v in preds.items()},  # predictions for each layer
                })

    df = pl.DataFrame(all_records)
    recordings_file_path = results_folder / f"predictions.csv"
    df.write_csv(recordings_file_path)

    targets = df.select('VernierOffset', 'Target')
    def format_col(layer):
        data = df.select('VernierOffset', 'GridPattern', layer)
        correct = data.with_columns(correct=(pl.col(layer) == targets))  # type: ignore
        agg_scores = correct.group_by('VernierOffset').mean()
        agg_scores.columns = ['VernierOffset', 'accuracy']
        return agg_scores.with_columns(pl.lit(scores.columns[-1]).alias("Layer"))

    scores = pl.concat([format_col(l) for l in df.columns[3:]])

    fig = px.line(scores, x='Layer', y='Accuracy')
    fig.write_image(results_folder / f"accuracy_vs_layer.png")

    _logger.info(f"Recording finished. Saved to: <{recordings_file_path}>")
    return recordings_file_path


def train_feature_extractor(feature_decoder, dataset):
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    trainer = Trainer(max_epochs=-1, max_steps=1000)
    trainer.fit(feature_decoder, train_dataloaders=dataloader)
    return feature_decoder


def record_all(annotations_file, models, model_names, results_folder):
    recording_paths = []
    for model, model_name in zip(models, model_names):

        feature_decoder = FeatureDecoder(
            model, target_dim=1, target_key='VernierType', loss='cross_entropy'
        )

        train_dataset = AnnotatedDataset(
            annotations_file,
            test_columns=TEST_COLUMNS,
            transform=model_transform(model),
            filter_expr="pl.col('VernierInOut') == 'outside'",
        )
        feature_decoder = train_feature_extractor(feature_decoder, train_dataset)

        test_dataset = AnnotatedDataset(
            annotations_file,
            test_columns=TEST_COLUMNS,
            transform=model_transform(model),
            filter_expr="pl.col('VernierInOut') == 'inside'",
        )
        dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        recording_paths.append(
            record_from_model((model_name, feature_decoder), dataloader, results_folder)
        )


def main(
    annotations_file,
    model_names,
    results_folder='',
    overwrite_recordings=False,
):
    _logger.info("Loading models...")

    results_folder = Path(results_folder) / 'amodal_completion'

    if not results_folder.exists() or overwrite_recordings:
        models = [init_model(m) for m in model_names]
        results_folder.mkdir(parents=True, exist_ok=True)
        _logger.info(f"Set results root folder to {results_folder}")
        record_all(annotations_file, models, model_names, results_folder)
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
        help="Overwrite the recording file if it already exists"
    )
    args = parser.parse_args()
    main(**vars(args))
