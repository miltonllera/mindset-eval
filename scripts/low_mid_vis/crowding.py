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


torch.set_float32_matmul_precision('high')


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

    feature_extractor = feature_extractor.eval().to(device=device)
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
                    **{
                        k: (v[i].sigmoid() > 0.5).to(dtype=torch.int).cpu().item()
                        for k, v in preds.items()},
                })

    df = pl.DataFrame(all_records)
    recordings_file_path = results_folder / f"predictions.csv"
    df.write_csv(recordings_file_path)

    targets = df['Target']
    def format_col(layer):
        data = df.select('VernierOffset', 'GridPattern', layer)
        data = data.with_columns(
            (pl.col(layer) == targets).alias(f'Correct?'),
            pl.col('GridPattern').map_elements(lambda x: len(x.split(','))).alias('Pattern Length'),
            pl.lit(layer).alias('Layer')
        )
        data = data.drop(layer, 'GridPattern')
        data = data.group_by('VernierOffset', 'Pattern Length', 'Layer').mean()
        data.columns = ['VernierOffset', 'Pattern Length', 'Layer', 'Accuracy']
        return data.sort(by=['VernierOffset', 'Pattern Length'])

    scores = pl.concat([format_col(l) for l in df.columns[4:]])

    fig = px.line(scores, x='Layer', y='Accuracy', color='Pattern Length')
    fig.write_image(results_folder / f"accuracy_vs_layer.png")

    _logger.info(f"Recording finished. Saved to: <{recordings_file_path}>")
    return recordings_file_path


def train_feature_extractor(feature_decoder, dataset):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=32)
    val_dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=32)
    trainer = Trainer(
        max_epochs=-1, max_steps=1000, val_check_interval=100, check_val_every_n_epoch=None
    )
    trainer.fit(feature_decoder, train_dataloaders=dataloader, val_dataloaders=val_dataloader)
    return feature_decoder


def record_all(annotations_file, model_names, results_folder):
    recording_paths = []
    for model_name in model_names:
        model = init_model(model_name)
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
    return recording_paths


def main(
    annotations_file,
    model_names,
    results_folder='',
    overwrite_recordings=False,
):
    _logger.info("Loading models...")

    results_folder = Path(results_folder) / 'un_crowding'

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
        help="Overwrite the recording file if it already exists"
    )
    args = parser.parse_args()
    main(**vars(args))
