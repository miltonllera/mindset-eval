import sys
import os
import os.path  as osp
import argparse
import logging
from functools import partial
from typing import Callable

import torch
import torchvision.transforms as transforms
import timm

sys.path.append("mindset/")
from mindset.src.utils.similarity_judgment.activation_recorder import RecordDistance


RESULTS_ROOT = "data/results/rel_vs_coord"

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


# init_model = partial(timm.create_model, pretrained=True)  # type: ignore
def init_model(model_name, device='auto', verbose=False):
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = timm.create_model(  # type: ignore
        model_name, pretrained=True, cache_dir="data/models/"
    ).to(device)

    if verbose: print(model)

    return model


def record_from_model(
    model: tuple[str, torch.nn.Module],
    metric: str,
    transform_fn: Callable,
    annotations_file: str,
    results_folder: str,
):
    model_name, net = model

    results_folder = os.path.join(results_folder, model_name)
    os.makedirs(results_folder, exist_ok=True)

    _logger.info(f"Recording from model: <{model_name}>")

    recorder = RecordDistance(
        annotations_file,
        match_factors=['Id'],
        non_match_factors=[],  # don't know what this should be
        factor_variable=['Class'],
        reference_level='basis',
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
            'scale': [1.0, 1.5],
            'rotation': [0, 360],
        },
        transformed_repetition=20,
        path_save_fig=results_folder,
        add_columns=[],
    )

    _logger.info(f"Recording finished. Results in: <{results_folder}>")
    distance_df.to_csv(osp.join(results_folder, "distance_df.csv"))

    return distance_df

def main(annotations_file, model_names, save_folder=''):
    models = [init_model(m) for m in model_names]

    if save_folder != '':
        results_folder = osp.join(RESULTS_ROOT, save_folder)
        os.makedirs(results_folder, exist_ok=True)
    else:
        results_folder = RESULTS_ROOT

    _logger.info(f"Set results root folder to {RESULTS_ROOT}")

    norm_values = dict(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    resize_value = 224

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
        transform_fn=transform_fn,
        results_folder=results_folder,
    )

    for m, n in zip(models, model_names):
        record((n, m))


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

    args = parser.parse_args()
    main(**vars(args))
