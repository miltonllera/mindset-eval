import torch
import torchvision.transforms as transforms
import timm


def init_model(model_name, verbose=False):
    model = timm.create_model(  # type: ignore
        model_name, pretrained=True, cache_dir="data/models/"
    ).to(get_device())

    if verbose:
        print(model)

    return model


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def model_transform(model) -> transforms.Compose:
    cfg = model.pretrained_cfg
    return transforms.Compose([
        transforms.Resize(cfg["input_size"][-1]),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
