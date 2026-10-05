import argparse
import shutil
from math import prod
from pathlib import Path

import torch
import polars as pl
import pyarrow.parquet as pq
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


record_from = [
  "^act1:in", "^layer[1-4].([0-9]|[1-2][0-9]).act3:in", "^fc:in",
]
TEST_COLUMNS = [
    'Path', 'VernierType', 'VernierOffset', 'GridPattern', 'GridArrangement',
    'NumRows', 'NumCols', 'ShapeSize', 'CenterShape', 'AlternateShape', 'Loc', 'Scale'
]
TARGET_COLUMN = 'VernierType'
annotations_file = "data/datasets/low_mid_vision/uncrowding_distributions/annotation.csv"
device = torch.device('mps')

def train_feature_extractor(feature_decoder, dataset):
    train_data, val_data = random_split(dataset, [0.8, 0.2])
    dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=0)
    trainer = Trainer(
        max_epochs=-1, max_steps=10, val_check_interval=5, check_val_every_n_epoch=None,
    )
    trainer.fit(feature_decoder, train_dataloaders=dataloader, val_dataloaders=val_dataloader)
    return feature_decoder


def test_recording():

    model = init_model('resnet50s.gluon_in1k').to(device=device)
    feature_extractor = FeatureDecoder(
        model, target_dim=1, target_key='VernierType', loss='cross_entropy',
        decode_from=record_from,
    ).to(device=device)

    available_cols = pl.read_csv(annotations_file, n_rows=1).columns
    test_columns = [c for c in TEST_COLUMNS if c in available_cols]

    train_dataset = AnnotatedDataset(
        annotations_file,
        test_columns=test_columns,
        transform=model_transform(model),
        filter_expr="pl.col('VernierInOut') == 'outside'",
    )

    feature_extractor = train_feature_extractor(feature_extractor, train_dataset)

    test_dataset = AnnotatedDataset(
        annotations_file,
        test_columns=test_columns,
        transform=model_transform(model),
        filter_expr="pl.col('VernierInOut') == 'inside'",
    )
    dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    buffer_chunks = []
    sample_counter = 0
    feature_extractor = feature_extractor.eval().to(device=device)
    writer = None

    def convert_if_tensor(data):
        for i in range(len(data)):
            if isinstance(data[i], torch.Tensor):
                data[i] = data[i].item()
        return data

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='test')):
            images = batch['Image'].to(device)
            preds = feature_extractor(images)
            targets = batch[TARGET_COLUMN]
            bsz = len(images)

            sample_ids = list(range(sample_counter, sample_counter + bsz))
            sample_counter += bsz

            chunk_dict = {k: convert_if_tensor(list(batch[k])) for k in TEST_COLUMNS if k in batch}
            chunk_dict['SampleID'] = sample_ids
            chunk_dict['Target'] = [int(t) if isinstance(t, torch.Tensor) else t for t in targets]

            for k, v in preds.items():
                pred_tensor = v.squeeze(-1) if v.ndim > 1 else v
                chunk_dict[k] = (pred_tensor.sigmoid() > 0.5).to(dtype=torch.int).cpu().tolist()

            print(chunk_dict)
            buffer_chunks.append(pl.DataFrame(chunk_dict))


test_recording()
