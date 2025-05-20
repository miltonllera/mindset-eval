import sys
import argparse
import logging
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import lightning.pytorch as pl
import lightning.pytorch.callbacks as callbacks
import timm
import pandas as pd
import sty
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.decoder import DecoderWrapper

sys.path.append("mindset/")
from mindset.src.utils.dataset_utils import get_dataloader
from mindset.src.utils.device_utils import set_global_device, to_global_device


RESULTS_ROOT = "data/results/tilt"

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


#---------------------------------------- Load models --------------------------------------------

def init_model(model_name, verbose=False):
    model = timm.create_model(model_name, pretrained=True, cache_dir="data/models/")  # type: ignore
    model = to_global_device(model)

    if verbose:
        print(model)

    return model


#--------------------------------------- Decoder Training ----------------------------------------

def train_decoders(
    model: nn.Module,
    target_dim: int,
    train_loader: DataLoader,
    train_epochs: int,
    save_path: Path,
):
    decoder_wrapper = DecoderWrapper(model, target_dim)
    checkpointer = callbacks.ModelCheckpoint(
        monitor="loss",        # metric to monitor
        filename="decoder_ckpt.pt",
        save_top_k=1,              # only keep best
        mode="min",
        dirpath=save_path,
    )
    trainer = pl.Trainer(max_epochs=train_epochs, callbacks=[checkpointer])
    trainer.fit(decoder_wrapper, train_loader)

    decoder_wrapper = decoder_wrapper.to(torch.device('cpu'))

    return decoder_wrapper


#---------------------------------------- Recording ----------------------------------------------

def evaluate_decoder(
    model: tuple[str, torch.nn.Module],
    annotations_file: str,
    results_folder: Path,
    retrain_decoder: bool = False,
):
    model_name, net = model

    results_folder = results_folder / model_name
    results_folder.mkdir(parents=True, exist_ok=True)
    _logger.info(f"Recording from model: <{model_name}>")

    # Load dataset
    transf_config={
        'values': {
            'translation_X': [-0.2, 0.2],
            'translation_Y': [-0.2, 0.2],
            'scale': [1.0, 1.2],
            'rotation': [0, 360],
        },
        'fill_color': (0, 0, 0),
        'size': net.pretrained_cfg['input_size'][-1],  # type: ignore
    }

    dataloader = get_dataloader(
        task_type='regression',
        ds_config={
            'name': "tilt_illusion",
            'annotation_file': annotations_file,
            'img_path_col_name': "Path",
            'label_cols': ["ThetaCenter"],
            'filters': {'Type': "center_context"},
            'neg_filters': {},
        },
        transf_config=transf_config,
        batch_size=100,
        return_path=True,
    )

    target_dim = next(iter(dataloader))[1].shape[-1]

    # train decoders if not available
    decoder_ckpt = results_folder / 'decoder_ckpt.pt.ckpt'
    if not decoder_ckpt.exists() or retrain_decoder:
        train_loader = get_dataloader(
            task_type='regression',
            ds_config={
                'name': "jastrow_illusion",
                'annotation_file': annotations_file,
                'img_path_col_name': "Path",
                'label_cols': ["ThetaCenter"],
                'filters': {'Type': "only_center"},
                'neg_filters': {},
            },
            transf_config=transf_config,
            batch_size=64,
            return_path=False,
        )

        decoder_wrapper = train_decoders(
            net, target_dim, train_loader,
            train_epochs=20,
            save_path=results_folder,
        )
    else:
        decoder_wrapper = DecoderWrapper.load_from_checkpoint(
            results_folder / 'decoder_ckpt.pt.ckpt',
            map_location=torch.device('cpu'),
            model=net,
            target_dim=target_dim,
        )

    # evaluate the decoders
    print(
        f"Evaluating Dataset "
        + sty.fg.green
        + f"{dataloader.dataset.name}"  # type: ignore
        + sty.rs.fg
    )

    results_final = []
    for _, data in enumerate(tqdm(dataloader, colour="yellow")):
        images, targets, path = data
        images = to_global_device(images)
        targets = to_global_device(targets)
        output = decoder_wrapper(images)

        for i in range(len(targets)):
            results_dict = {
                'image_path': path[i],
                'target_size_top': targets[i][0].item(),
                'target_size_bottom': targets[i][1].item(),
                **{
                    f"prediction_size_top_dec_{dec_idx}": output[dec_idx][i][0].item()
                    for dec_idx in range(len(decoder_wrapper.decoders))
                },
                **{
                    f"prediction_size_bottom_dec_{dec_idx}": output[dec_idx][i][1].item()
                    for dec_idx in range(len(decoder_wrapper.decoders))
                },
            }

            results_final.append(results_dict)

    results_final_pandas = pd.DataFrame(results_final)
    results_final_pandas.to_csv(results_folder / "predictions.csv", index=False)

    return results_folder


def evaluate_all(annotations_file, models, model_names, results_folder):
    record = partial(
        evaluate_decoder,
        annotations_file=annotations_file,
        results_folder=results_folder,
    )

    recording_paths = []
    for m, n in zip(models, model_names):
        recording_paths.append(record((n, m)))

    return recording_paths


def main(
    annotations_file,
    model_names,
    overwrite_recordings=False,
):
    _logger.info("Loading models...")

    results_folder = Path(RESULTS_ROOT)

    device = 'cpu'
    set_global_device(device)

    if not results_folder.exists() or overwrite_recordings:
        models = [init_model(m) for m in model_names]
        results_folder.mkdir(parents=True, exist_ok=True)
        _logger.info(f"Set results root folder to {RESULTS_ROOT}")
        evaluate_all(annotations_file, models, model_names, results_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", type=str, nargs='+', dest='model_names',
        help="List of models to test"
    )
    parser.add_argument("--annotations_file", type=str,
        help="Path to the annotations file used to run the experiment."
    )
    parser.add_argument("--overwrite_recordings", action='store_true',
        help="Overwrite the recording file if it all already exists"
    )

    args = parser.parse_args()
    main(**vars(args))




