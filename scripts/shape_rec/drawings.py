import sys
import argparse
import logging
from functools import partial
from pathlib import Path

import numpy as np
import torch
import timm
import pandas as pd
import sty
from tqdm import tqdm

sys.path.append("mindset/")
from mindset.src.utils.dataset_utils import get_dataloader, ImageNetClasses
from mindset.src.utils.device_utils import set_global_device, to_global_device


RESULTS_ROOT = "data/results/drawings"

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
    dataset_name,
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
            'name': dataset_name,
            'annotation_file': annotations_file,
            'img_path_col_name': "Path",
            'label_cols': "Class",
            'filters': [],
        },
        transf_config={
            'values': {
                'translation_X': [-0.2, 0.2],
                'translation_Y': [-0.2, 0.2],
                'scale': [1.0, 1.2],
                'rotation': [0, 360],
            },
            'fill_color': (0, 0, 0),
        },
        batch_size=10,
        return_path=True,
    )
    imagenet_classes = ImageNetClasses(class_index_json_file="mindset/assets/imagenet_class_index.json")

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


def evaluate_all(dataset_name, annotations_file, models, model_names, results_folder):
    record = partial(
        evaluate_model,
        dataset_name=dataset_name,
        annotations_file=annotations_file,
        results_folder=results_folder,
    )

    recording_paths = []
    for m, n in zip(models, model_names):
        recording_paths.append(record((n, m)))

    return recording_paths


def main(
    drawing_type: str,
    annotations_file,
    model_names,
    overwrite_recordings=False,
):
    _logger.info("Loading models...")

    results_folder = Path(RESULTS_ROOT) / drawing_type

    device = 'cpu'
    set_global_device(device)

    if not results_folder.exists() or overwrite_recordings:
        models = [init_model(m) for m in model_names]
        results_folder.mkdir(parents=True, exist_ok=True)
        _logger.info(f"Set results root folder to {RESULTS_ROOT}")
        evaluate_all(drawing_type, annotations_file, models, model_names, results_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--models", type=str, nargs='+', dest='model_names',
        help="List of models to test"
    )
    parser.add_argument("--drawing_type", type=str,
        choices=['line', 'dotted', 'silhouettes', 'texture_lines', 'texture_chars'],
        help="Type of drawing"
    )
    parser.add_argument("--annotations_file", type=str,
        help="Path to the annotations file used to run the experiment."
    )
    parser.add_argument("--overwrite_recordings", action='store_true',
        help="Overwrite the recording file if it all already exists"
    )

    args = parser.parse_args()
    main(**vars(args))

