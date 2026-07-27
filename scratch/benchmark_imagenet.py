import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, ImageNet
from tqdm import tqdm

# Ensure project root is in sys.path when script is executed from subdirectories or scratch
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import get_device, init_model, model_transform


def get_dataset(data_dir: str | Path, split: str, transform):
    """Load ImageNet dataset for a given split using torchvision utilities.
    
    Tries torchvision.datasets.ImageNet first. If raw archives aren't present or extracted,
    falls back to torchvision.datasets.ImageFolder on data_dir/split directory layout.
    """
    data_dir = Path(data_dir)
    split_name = "val" if split in ["val", "test"] else "train"

    try:
        # Try PyTorch's built-in ImageNet dataset class
        dataset = ImageNet(root=str(data_dir), split=split_name, transform=transform)
    except (RuntimeError, FileNotFoundError):
        # Fallback to ImageFolder structure (data_dir/train or data_dir/val)
        split_dir = data_dir / split_name
        if not split_dir.exists():
            split_dir = data_dir
        dataset = ImageFolder(root=str(split_dir), transform=transform)

    return dataset


@torch.no_grad()
def evaluate_split(model, dataloader, device, max_samples: int | None = None) -> tuple[float, float]:
    """Evaluate model accuracy on a dataloader.
    
    Returns:
        (top1_acc, top5_acc) as percentages (0.0 to 100.0).
    """
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0

    for images, targets in tqdm(dataloader, desc="Evaluating", leave=False):
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)

        # Get top-5 predictions
        _, pred = outputs.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        correct_top1 += correct[:1].reshape(-1).float().sum(0).item()
        correct_top5 += correct[:5].reshape(-1).float().sum(0).item()
        total_samples += targets.size(0)

        if max_samples and total_samples >= max_samples:
            break

    top1_acc = (correct_top1 / total_samples) * 100.0
    top5_acc = (correct_top5 / total_samples) * 100.0
    return top1_acc, top5_acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate a timm model on ImageNet dataset.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet18",
        help="Name of the model from PyTorch Image Models (timm). Default: resnet18",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/datasets/imagenet",
        help="Path to ImageNet dataset directory. Default: data/datasets/imagenet",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation. Default: 64",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers. Default: 4",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate per split (useful for quick dry runs).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed model architecture.",
    )

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    print(f"Initializing model '{args.model_name}' using init_model...")
    model = init_model(args.model_name, verbose=args.verbose)
    transform = model_transform(model)

    print(f"Loading ImageNet dataset from '{args.data_dir}'...")

    # Train split
    train_dataset = get_dataset(args.data_dir, split="train", transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Test/Val split
    test_dataset = get_dataset(args.data_dir, split="val", transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train split size: {len(train_dataset)} samples")
    print(f"Test/Val split size: {len(test_dataset)} samples")

    print("\n--- Evaluating Training Split ---")
    train_top1, train_top5 = evaluate_split(
        model, train_loader, device, max_samples=args.max_samples
    )
    print(f"Training Accuracy -> Top-1: {train_top1:.2f}%, Top-5: {train_top5:.2f}%")

    print("\n--- Evaluating Test/Validation Split ---")
    test_top1, test_top5 = evaluate_split(
        model, test_loader, device, max_samples=args.max_samples
    )
    print(f"Test Accuracy     -> Top-1: {test_top1:.2f}%, Top-5: {test_top5:.2f}%")


if __name__ == "__main__":
    main()
