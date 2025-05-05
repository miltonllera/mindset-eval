import sys
import argparse
import logging
from functools import partial
from pathlib import Path

import torch
import timm
import torchvision.transforms as transforms

sys.path.append("mindset/")
from mindset.src.utils.similarity_judgment.activation_recorder import RecordDistance
from mindset.src.utils.device_utils import set_global_device, to_global_device
from scripts.analysis import  get_recording_files


RESULTS_ROOT = "data/results/amodal_completion"

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
    annotations_file: str,
    results_folder: Path,
):
    model_name, net = model

    transform_fn = transforms.Compose([
        transforms.Resize(net.pretrained_cfg['input_size'][-1]),  # type: ignore
        transforms.ToTensor(),
        transforms.Normalize(net.pretrained_cfg['mean'], net.pretrained_cfg['std'])  # type: ignore
    ])

    results_folder = results_folder / model_name
    results_folder.mkdir(parents=True, exist_ok=True)

    _logger.info(f"Recording from model: <{model_name}>")

    recorder = RecordDistance(
        annotations_file,
        factor_variable='Type',
        reference_level='no_occlusion',
        match_factors=['TopShape', 'SampleId'],
        non_match_factors=[],  # don't know what this should be
        filter_factor_level={},
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
            'scale': [1.0, 1.2],
            'rotation': [0, 360],
        },
        transformed_repetition=20,
        path_save_fig=results_folder,
        add_columns=[],
    )

    _logger.info(f"Recording finished. Figures in: <{results_folder}>")
    recordings_file_path = results_folder / f"{metric}.csv"
    distance_df.to_csv(recordings_file_path)

    return recordings_file_path


def record_all(annotations_file, models, model_names, results_folder):
    record = partial(
        record_from_model,
        metric= "cossim",
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
    save_folder='',
    overwrite_recordings=False,
    comparison_levels=None,
):
    _logger.info("Loading models...")

    results_folder = Path(RESULTS_ROOT)
    if save_folder != '':
        results_folder = results_folder / save_folder

    device = 'cpu'
    set_global_device(device)

    if not results_folder.exists() or overwrite_recordings:
        models = [init_model(m) for m in model_names]
        results_folder.mkdir(parents=True, exist_ok=True)
        _logger.info(f"Set results root folder to {RESULTS_ROOT}")
        recording_files = record_all(annotations_file, models, model_names, results_folder)
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
