import sys
import argparse
import logging
from functools import partial
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.optim  as optim
import timm
import pandas as pd
import sty
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.decoder import DecoderWrapper

sys.path.append("mindset/")
from mindset.src.utils.dataset_utils import get_dataloader
from mindset.src.utils.device_utils import set_global_device, to_global_device


RESULTS_ROOT = "data/results/ebbinghaus"

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
    learning_rate: float,
    train_epochs: int,
    save_path: Path,
):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    decoder_wrapper = DecoderWrapper(model, target_dim).to(device)

    optimizers = []
    for dec in decoder_wrapper.decoders.values():
        o = optim.Adam(dec.parameters(), lr=learning_rate)
        optimizers.append(o)

    loss_fn = nn.MSELoss()

    for _ in range(train_epochs):
        pbar = tqdm(train_loader, desc="Decoder Training")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            decoder_preds = decoder_wrapper(x)

            for pred, o in zip(decoder_preds, optimizers):
                loss = loss_fn(pred, y)
                o.zero_grad()
                loss.backward()
                o.step()

    decoder_wrapper = decoder_wrapper.to(torch.device('cpu'))
    torch.save(decoder_wrapper.state_dict(), save_path)

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
            'name': "ebbinghaus_illusion",
            'annotation_file': annotations_file,
            'img_path_col_name': "Path",
            'label_cols': "NormSizeCenterCircle",
            'filters': {'Category': "scrambled_circles"},
        },
        transf_config=transf_config,
        batch_size=100,
        return_path=True,
    )

    target_dim = next(iter(dataloader))[1].shape[-1]

    # train decoders if not available
    decoder_ckpt = results_folder / 'decoder_ckpt.pt'
    if not decoder_ckpt.exists() or retrain_decoder:
        train_loader = get_dataloader(
            task_type='regression',
            ds_config={
                'name': "ebbinghaus_illusion",
                'annotation_file': annotations_file,
                'img_path_col_name': "Path",
                'label_cols': "NormSizeCenterCircle",
                'filters': {},
                'neg_filters': {'Category': "scrambled_circles"},
            },
            transf_config=transf_config,
            batch_size=64,
            return_path=False,
        )

        decoder_wrapper = train_decoders(
            net, target_dim, train_loader,
            learning_rate=1e-5,
            train_epochs=20,
            save_path=decoder_ckpt,
        )
    else:
        decoder_wrapper = DecoderWrapper(net, target_dim)
        decoder_wrapper.load_state_dict(torch.load(decoder_ckpt))


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
            for i in range(len(targets)):
                results_final.append({
                        "image_path": path[i],
                        "label": targets[i].item(),
                        **{
                            f"prediction_dec_{dec_idx}": (output[dec_idx][i].item())
                            for dec_idx in range(len(decoder_wrapper.decoders))
                        },
                    }
                )

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
