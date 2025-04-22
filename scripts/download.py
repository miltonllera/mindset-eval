import os
import argparse


def main(version: str, path: str, overwrite: bool):
    dataset = 'minsdset'
    if version == "lite":
        dataset = 'mindset-lite'
    elif version == "full":
        dataset = 'mindset'
    else:
        raise RuntimeError()

    path = f"data/datasets/{dataset}"
    if os.path.exists(path) and not overwrite:
        raise ValueError("Dataset path already exists, set overwrite flag to download anyway")

    os.system(
        f"curl -L -o data/datasets/{version}/mindset.zip" \
        "https://www.kaggle.com/api/v1/datasets/download/mindsetvision/{dataset}"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Download the mindset dataset")

    parser.add_argument("--version", type=str, choices=['full', 'lite'], default='lite',
        help="Version of the dataset to download"
    )
    parser.add_argument("--path", type=str, default='data/dataset',
        help="Path used to save the downloaded dataset."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the dataset if it already exists",
    )

    args = parser.parse_args()

    main(**vars(args))
