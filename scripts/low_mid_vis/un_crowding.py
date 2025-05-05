import sys
import argparse
import logging
from functools import partial
from pathlib import Path
from typing import Callable

import torch
import timm
import torchvision.transforms as transforms

sys.path.append("mindset/")
from mindset.src.utils.similarity_judgment.activation_recorder import RecordDistance
from mindset.src.utils.device_utils import set_global_device, to_global_device
from scripts.analysis import get_recording_files


RESULTS_ROOT = "data/results/crowding"

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


#---------------------------------------- Load models --------------------------------------------

def init_model(model_name, verbose=False):
    model = timm.create_model(model_name, pretrained=True, cache_dir="data/models/")  # type: ignore
    model = to_global_device(model)

    if verbose:
        print(model)

    return model


#---------------------------------------- Recording ----------------------------------------------

def record_from_model(
    model: tuple[str, torch.nn.Module],
    metric: str,
    transform_fn: Callable,
    annotations_file: str,
    experiment_type: str,
    results_folder: Path,
):
    model_name, net = model

    results_folder = results_folder / model_name
    results_folder.mkdir(parents=True, exist_ok=True)

    _logger.info(f"Recording from model: <{model_name}>")

    recorder = RecordDistance(
        annotations_file,
        factor_variable='ShapeCode',
        reference_level='none',
        match_factors=[],
        non_match_factors=[],
        filter_factor_level={'VenierInOut': experiment_type},
        distance_metric=metric,
        net=net,
        only_save=["Conv2d", "Linear"],
    )

    distance_df, layer_names = recorder.compute_from_annotation(
        transform_fn,
        matching_transform=True,
        fill_bk=[0, 0, 0],
        transf_boundaries={  # type: ignore
            'translation_X': [-0.2, 0.2],
            'translation_Y': [-0.2, 0.2],
            'scale': [1.0, 1.5],
            'rotation': [0, 360],
        },
        transformed_repetition=5,
        path_save_fig=results_folder,
        add_columns=[],
    )

    _logger.info(f"Recording finished. Figures in: <{results_folder}>")
    recordings_file_path = results_folder / f"{metric}.csv"
    distance_df.to_csv(recordings_file_path)

    return recordings_file_path


def record_all(annotations_file, experiment_type, models, model_names, results_folder):
    norm_values = dict(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    resize_value = 224

    if experiment_type  not in ['crowding', 'uncrowding']:
        raise RuntimeError("experiment_type must be either 'crowding' or 'uncrowding'")

    # get_user_attributes(model)
    transform_fn = transforms.Compose([
        transforms.Resize(resize_value),
        transforms.ToTensor(),
        transforms.Normalize(norm_values['mean'], norm_values['std'])
    ])

    record = partial(
        record_from_model,
        metric= "cossim",
        annotations_file=annotations_file,
        experiment_type=experiment_type,
        transform_fn=transform_fn,
        results_folder=results_folder,
    )

    recording_paths = []
    for m, n in zip(models, model_names):
        recording_paths.append(record((n, m)))

    return recording_paths


def main(
    annotations_file,
    experiment_type,
    model_names,
    save_folder='',
    overwrite_recordings=False,
    comparison_levels=None,
):
    _logger.info("Loading models...")

    results_folder = Path(RESULTS_ROOT)
    if save_folder != '':
        results_folder = results_folder / save_folder

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    set_global_device(device)

    if not results_folder.exists() or overwrite_recordings:
        models = [init_model(m) for m in model_names]
        results_folder.mkdir(parents=True, exist_ok=True)
        _logger.info(f"Set results root folder to {RESULTS_ROOT}")
        recording_files = record_all(
            annotations_file, experiment_type, models, model_names, results_folder
        )
    else:
        recording_files = get_recording_files(results_folder, model_names)

    # analyize_all(recording_files, comparison_levels, start_from=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", type=str, nargs='+', dest='model_names',
        help="List of models to test"
    )
    parser.add_argument("--annotations_file", type=str,
        help="Path to the annotations file used to run the experiment."
    )
    parser.add_argument("--type", choices=['crowding', 'uncrowding'], dest="experiment_type",
        default='crowding', help="Whether to test crowding or uncrowding"
    )
    parser.add_argument("--save_folder", type=str, default='',
        help="Experiment folder where to store all results"
    )
    parser.add_argument("--overwrite_recordings", action='store_true',
        help="Overwrite the recording file if it all already exists"
    )
    parser.add_argument("--comparison_levels", type=str, nargs="+", default=None,
        help=""
    )

    args = parser.parse_args()
    main(**vars(args))
