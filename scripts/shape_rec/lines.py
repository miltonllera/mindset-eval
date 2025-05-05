import sys
import argparse
import logging
from functools import partial
from pathlib import Path

import numpy as np
import torch
import timm
import torchvision.transforms as transforms
import pandas as pd
import sty
from tqdm import tqdm

sys.path.append("mindset/")
from mindset.src.utils.dataset_utils import get_dataloader, ImageNetClasses
from mindset.src.utils.device_utils import set_global_device, to_global_device
from scripts.analysis import  get_recording_files


RESULTS_ROOT = "data/results/rel_vs_coord"

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

def evaluate_model(
    model: tuple[str, torch.nn.Module],
    metric: str,
    annotations_file: str,
    results_folder: Path,
):
    model_name, net = model

    results_folder = results_folder / model_name
    results_folder.mkdir(parents=True, exist_ok=True)

    _logger.info(f"Recording from model: <{model_name}>")

    dataloader = get_dataloader(
        task_type='classification',
        ds_config={
            'name': "line-drawings",
            'annotations_file': annotations_file,
            'img_path_col_name': "Path",
            'label_cols': ...,
            'filters': [],
        },
        transf_config=None,
        batch_size=10,
        return_path=False,
    )
    imagenet_classes = ImageNetClasses()

    results_final = []
    results_path = results_folder / dataloader.dataset.name  # type: ignore
    results_path.mkdir(parents=True, exist_ok=True)

    print(
        f"Evaluating Dataset "
        + sty.fg.green
        + f"{dataloader.dataset.name}"  # type: ignore
        + sty.rs.fg
    )

    for _, data in enumerate(tqdm(dataloader, colour="yellow")):
        images, labels, path = data
        images = to_global_device(images)
        labels = to_global_device(labels)
        output = net(images)
        for i in range(len(labels)):
            # Top 5 prediction
            prediction = torch.topk(output[i], 5).indices.tolist()

            results_final.append(
                {
                    "image_path": path[i],
                    "label_idx": labels[i].item(),
                    "label_class_name": imagenet_classes.idx2label[
                        labels[i].item()
                    ],
                    **{f"prediction_idx_top_{i}": prediction[i] for i in range(5)},
                    **{
                        f"prediction_class_name_top_{i}": imagenet_classes.idx2label[
                            prediction[i]
                        ]
                        for i in range(5)
                    },
                    "Top-5 At Least One Correct": np.any(
                        [labels[i].item() in prediction]
                    ),
                }
            )

    results_final_pandas = pd.DataFrame(results_final)
    results_final_pandas.to_csv(results_path / "predictions.csv", index=False)

    top_5_accuracy = np.mean(
        [
            results_final_pandas["label_idx"][i]
            in list(
                results_final_pandas[
                    [
                        "prediction_idx_top_0",
                        "prediction_idx_top_1",
                        "prediction_idx_top_2",
                        "prediction_idx_top_3",
                        "prediction_idx_top_4",
                    ]
                ].iloc[i]
            )
            for i in range(len(results_final_pandas))
        ]
    )
    print(
        f"Accuracy: {np.mean(results_final_pandas['label_idx'] == results_final_pandas['prediction_idx_top_0'])}"
    )
    print(f"Top 5 Accuracy: {top_5_accuracy}")

    return results_path


def evaluate_all(annotations_file, models, model_names, results_folder):
    record = partial(
        evaluate_model,
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
):
    _logger.info("Loading models...")

    results_folder = Path(RESULTS_ROOT)
    if save_folder != '':
        results_folder = results_folder / save_folder

    if save_folder != '':
        results_folder = results_folder / save_folder

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
    parser.add_argument("--save_folder", type=str, default='',
        help="Experiment folder where to store all results"
    )

    args = parser.parse_args()
    main(**vars(args))

