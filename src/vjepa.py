"""V-JEPA 2.1 timm feature encoders for MindSet still-image benchmarks.

Use ``src.utils.init_model`` or call ``register()`` before ``timm.create_model``.
The pinned Meta source and official weights are downloaded on first use, then
reused offline. Set MINDSET_VJEPA_MODEL_DIR to reuse an existing model cache.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import sys
import tarfile
import tempfile
import threading
import types
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from timm.models import is_model, model_entrypoint, register_model

UPSTREAM_REVISION = "204698b45b3712590f06245fbfba32d3be539812"
SOURCE_URL = (
    f"https://codeload.github.com/facebookresearch/vjepa2/tar.gz/{UPSTREAM_REVISION}"
)
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/vjepa2/"
MODEL_NAMES = ("vjepa2_1_vit_base_384", "vjepa2_1_vit_large_384")
_CHECKPOINTS = {
    MODEL_NAMES[0]: (
        "vjepa2_1_vitb_dist_vitG_384.pt",
        "848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d",
    ),
    MODEL_NAMES[1]: (
        "vjepa2_1_vitl_dist_vitG_384.pt",
        "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6",
    ),
}
_IMPORT_LOCK = threading.RLock()
_UPSTREAM_MODULES: dict[Path, dict[str, types.ModuleType]] = {}


def _is_upstream_namespace(name: str) -> bool:
    return name in ("src", "app") or name.startswith(("src.", "app."))


@contextmanager
def _upstream_imports(source: Path) -> Iterator[types.ModuleType]:
    """Temporarily isolate Meta's src/app packages from MindSet's src package.

    Construction is serialized, but callers must not import src/app in another
    thread while building an encoder. Forward passes need no namespace changes.
    """
    source = source.resolve()
    with _IMPORT_LOCK:
        saved = {
            name: mod
            for name, mod in sys.modules.copy().items()
            if _is_upstream_namespace(name)
        }
        ready = False
        try:
            for name in saved:
                sys.modules.pop(name)
            cached = _UPSTREAM_MODULES.get(source)
            if cached is not None:
                sys.modules.update(cached)
            else:
                for name in ("src", "app"):
                    package_dir = source / name
                    init_file = package_dir / "__init__.py"
                    if init_file.is_file():
                        spec = importlib.util.spec_from_file_location(
                            name,
                            init_file,
                            submodule_search_locations=[str(package_dir)],
                        )
                        assert spec is not None
                    else:
                        # The pinned revision uses namespace packages.
                        spec = importlib.machinery.ModuleSpec(
                            name, loader=None, is_package=True
                        )
                        spec.submodule_search_locations = [str(package_dir)]
                    package = importlib.util.module_from_spec(spec)
                    sys.modules[name] = package
                    if init_file.is_file():
                        assert spec.loader is not None
                        spec.loader.exec_module(package)
            backbones = importlib.import_module("src.hub.backbones")
            ready = True
            yield backbones
        finally:
            imported = {
                name: mod
                for name, mod in sys.modules.copy().items()
                if _is_upstream_namespace(name)
            }
            if ready:
                _UPSTREAM_MODULES[source] = imported
            for name in imported:
                sys.modules.pop(name)
            sys.modules.update(saved)


def _model_dir(model_dir: str | Path | None, cache_dir: str | Path | None) -> Path:
    configured = model_dir or os.environ.get("MINDSET_VJEPA_MODEL_DIR")
    if configured is not None:
        return Path(configured).expanduser().resolve()
    if cache_dir is not None:
        return (Path(cache_dir).expanduser() / "vjepa2").resolve()
    return Path(__file__).resolve().parents[1] / "data/models/vjepa2"


def _download_source(directory: Path) -> Path:
    source = directory / f"vjepa2-{UPSTREAM_REVISION}"
    if not source.exists():
        directory.mkdir(parents=True, exist_ok=True)
        # Extract into a private directory so failed downloads/extraction cannot
        # leave a partial source tree in the reusable cache.
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            archive = Path(temporary) / "source.tar.gz"
            torch.hub.download_url_to_file(SOURCE_URL, str(archive))
            with tarfile.open(archive) as bundle:
                bundle.extractall(temporary, filter="data")
            extracted = Path(temporary) / source.name
            if not (extracted / "src/hub/backbones.py").is_file():
                raise FileNotFoundError(f"Missing V-JEPA backbone in {SOURCE_URL}")
            if not source.exists():
                extracted.rename(source)
    return source


def _image_forward(self, images: torch.Tensor, *args, **kwargs):
    """Use Meta's single-frame image embedding without renaming block hooks."""
    if images.ndim == 4:
        images = images.unsqueeze(2)
    elif images.ndim != 5 or images.shape[2] != 1:
        raise ValueError("V-JEPA MindSet entrypoints expect BCHW or BC1HW still images")
    return self._vjepa_forward(images, *args, **kwargs)


def _build_model(
    name: str,
    *,
    pretrained: bool,
    cache_dir=None,
    pretrained_cfg=None,
    pretrained_cfg_overlay=None,
    model_dir=None,
    upstream_dir=None,
    num_classes=0,
    in_chans=3,
    features_only=False,
    **kwargs,
):
    if features_only:
        raise ValueError("Use block hooks; features_only is not supported for V-JEPA")
    if num_classes != 0 or in_chans != 3:
        raise ValueError(
            "V-JEPA entrypoints are RGB feature encoders (num_classes=0, in_chans=3)"
        )
    forbidden = {
        "img_size",
        "patch_size",
        "tubelet_size",
        "num_frames",
        "img_temporal_dim_size",
    } & kwargs.keys()
    if forbidden:
        raise ValueError(
            f"Pinned V-JEPA still-image geometry cannot be overridden: {sorted(forbidden)}"
        )
    if pretrained_cfg is not None and not isinstance(pretrained_cfg, dict):
        raise ValueError("pretrained_cfg must be a dict, not a remote pretrained tag")

    directory = _model_dir(model_dir, cache_dir)
    source = (
        Path(upstream_dir).expanduser().resolve()
        if upstream_dir
        else _download_source(directory)
    )
    if source.name != f"vjepa2-{UPSTREAM_REVISION}":
        raise ValueError(
            f"V-JEPA source must be the pinned vjepa2-{UPSTREAM_REVISION} tree"
        )
    if not (source / "src/hub/backbones.py").is_file():
        raise FileNotFoundError(f"Missing pinned V-JEPA source: {source}")
    filename, sha256 = _CHECKPOINTS[name]
    checkpoint = directory / filename
    if pretrained and not checkpoint.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        # PyTorch streams to a temporary file, verifies the hash, then moves it
        # into place. Never use the localhost hub URL in this upstream revision.
        torch.hub.download_url_to_file(
            CHECKPOINT_URL + filename, str(checkpoint), hash_prefix=sha256
        )

    with _upstream_imports(source) as backbones:
        encoder, predictor = getattr(backbones, name)(pretrained=False, **kwargs)
        del predictor
        if pretrained:
            state = torch.load(
                checkpoint, map_location="cpu", weights_only=True, mmap=True
            )
            encoder_state = backbones._clean_backbone_key(state["ema_encoder"])
            encoder.load_state_dict(encoder_state, strict=True)
            del encoder_state, state

    # Adapt only the root forward: wrapping the whole encoder would change
    # blocks.N to encoder.blocks.N and break existing layer selectors.
    encoder._vjepa_forward = encoder.forward
    encoder.forward = types.MethodType(_image_forward, encoder)
    cfg = {
        "architecture": name,
        "input_size": (3, 384, 384),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "interpolation": "bilinear",
        "crop_pct": 1.0,
        "fixed_input_size": False,
        "num_classes": 0,
        "first_conv": "patch_embed_img.proj",
        "classifier": "",
        "file": str(checkpoint),
    }
    for overrides in (pretrained_cfg, pretrained_cfg_overlay):
        if overrides:
            cfg.update(overrides)
    if tuple(cfg["input_size"]) != (3, 384, 384):
        raise ValueError(
            "V-JEPA MindSet pretrained_cfg input_size must remain (3, 384, 384)"
        )
    encoder.pretrained_cfg = cfg
    encoder.default_cfg = cfg.copy()
    encoder.num_classes = 0
    encoder.vjepa_upstream_revision = UPSTREAM_REVISION
    encoder.vjepa_checkpoint = str(checkpoint) if pretrained else None
    return encoder


def vjepa2_1_vit_base_384(pretrained: bool = False, **kwargs):
    """V-JEPA 2.1 distilled ViT-B/16 image encoder (12 blocks, 384 px)."""
    return _build_model(MODEL_NAMES[0], pretrained=pretrained, **kwargs)


def vjepa2_1_vit_large_384(pretrained: bool = False, **kwargs):
    """V-JEPA 2.1 distilled ViT-L/16 image encoder (24 blocks, 384 px)."""
    return _build_model(MODEL_NAMES[1], pretrained=pretrained, **kwargs)


def register() -> tuple[str, str]:
    """Register idempotently, refusing unrelated timm name collisions."""
    for entrypoint in (vjepa2_1_vit_base_384, vjepa2_1_vit_large_384):
        name = entrypoint.__name__
        entrypoint.__dict__["_mindset_vjepa_revision"] = UPSTREAM_REVISION
        if is_model(name):
            if (
                getattr(model_entrypoint(name), "_mindset_vjepa_revision", None)
                != UPSTREAM_REVISION
            ):
                raise RuntimeError(f"A different timm entrypoint already owns {name}")
        else:
            register_model(entrypoint)
    return MODEL_NAMES
