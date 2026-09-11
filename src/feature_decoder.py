import re
import logging
from typing import Callable, Literal
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics import Accuracy, R2Score
from src.layer_spec import (
    resolve_layer_targets,
    extract_stream_tensor,
    sanitize_key_for_module_dict,
)


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


class FeatureDecoder(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        target_dim: int,
        target_key: str,
        loss: Literal['mse', 'cross_entropy'] | Callable,
        decode_from: list[str] | None = None,
        finetune_model: bool = False
    ) -> None:
        super().__init__()

        if loss == 'mse':
            loss = nn.MSELoss()
        elif loss == 'cross_entropy':
            loss = nn.BCEWithLogitsLoss() if target_dim == 1 else nn.CrossEntropyLoss()

        self.decode_from = decode_from if decode_from is not None else ["Linear", "Conv2d"]
        self.net = net
        self.target_dim = target_dim
        self.target_key = target_key
        self.finetune_model = finetune_model
        self.loss = loss
        self._hooks = []
        self._decoders = nn.ModuleDict()
        self._extracted_features = {}
        self._register_hooks()
        self._init_decoders()

    def forward(self, x):
        self._extracted_features.clear()
        if not self.finetune_model:
            with torch.no_grad():
                self.net(x)
        else:
            self.net(x)

        decoder_preds = {}
        for k, v in self._extracted_features.items():
            if len(v.shape) > 2:
                v = v.flatten(1)
            safe_k = sanitize_key_for_module_dict(k)
            decoder_preds[k] = self._decoders[safe_k](v)

        self._extracted_features.clear()
        return decoder_preds

    def _step(self, batch, stage):  # type: ignore
        x, y = batch['Image'], batch[self.target_key]
        y = self._format_input(y)
        decoder_preds = self.forward(x)

        total_loss, decoder_losses = 0.0, {}
        for k, v in decoder_preds.items():
            decoder_losses[f'{stage}-loss/layer {k}'] = self.loss(v, y)
            total_loss += decoder_losses[f'{stage}-loss/layer {k}']

        self.log_dict(
            decoder_losses,
            prog_bar=True,
            on_step=stage == 'train',
            on_epoch=stage != 'train',
        )

        return total_loss, decoder_losses, decoder_preds

    def training_step(self, batch):
        return self._step(batch, 'train')[0]

    def validation_step(self, batch):
        total_loss, _, predictions = self._step(batch, 'val')

        if isinstance(self.loss, nn.BCEWithLogitsLoss):
            metric = Accuracy(task='binary').to(device=list(predictions.values())[0].device)
            name = 'acc'
        elif isinstance(self.loss, nn.CrossEntropyLoss):
            metric = Accuracy(task='multiclass')
            name = 'acc'
        else:
            metric = R2Score()
            name = 'r2'

        decoder_acc = {}
        for k, v in predictions.items():
            decoder_acc[f'val-{name}/layer {k}'] = metric(
                v, self._format_input(batch[self.target_key])
            )

        self.log_dict(
            decoder_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return total_loss

    def configure_optimizers(self):
        parameters = [p for p in self._decoders.parameters()]
        if self.finetune_model:
            parameters = [p for p in self.net.parameters()]
        return torch.optim.Adam(parameters, lr=1e-3)

    def _format_input(self, target):
        if len(target.shape) == 1:
            target = target[..., None]
        if isinstance(self.loss, nn.BCEWithLogitsLoss):
            target = target.to(dtype=torch.float32)
        return target

    def _init_decoders(self):
        input_size = self.net.pretrained_cfg['input_size']  # type: ignore
        sample_input = torch.randn(input_size)[None]  # type: ignore
        self.net.cpu()(sample_input)

        for k, v in self._extracted_features.items():
            if len(v.shape) > 2:
                feature_shape = v.numel()
            else:
                feature_shape = v.shape[-1]

            safe_k = sanitize_key_for_module_dict(k)
            self._decoders.add_module(
                safe_k,
                nn.Sequential(
                    nn.BatchNorm1d(feature_shape),
                    nn.Linear(feature_shape, 64),
                    nn.ELU(),
                    nn.Linear(64, 1)
                )
            )

        self._extracted_features.clear()

    def _register_hooks(self):
        targets = resolve_layer_targets(self.net, self.decode_from)
        for name, layer, stream in targets:
            self._hooks.append(layer.register_forward_hook(self._make_hook(name, stream)))

    def _make_hook(self, name: str, stream: str):
        def hook(_module, args, output):
            val = extract_stream_tensor(args, output, stream, layer_name=name)
            self._extracted_features[name] = (val.detach() if not self.finetune_model else val)
        return hook

def flatten_model(modules):
    def flatten_list(_2d_list):
        flat_list = []
        # Iterate through the outer list
        for element in _2d_list:
            if type(element) is list:
                # If the element is of type list, iterate through the sublist
                for item in element:
                    flat_list.append(item)
            else:
                flat_list.append(element)
        return flat_list

    ret = []
    try:
        for _, n in modules:
            ret.append(flatten_model(n))
    except:
        try:
            if str(modules._modules.items()) == "odict_items([])":
                ret.append(modules)
            else:
                for _, n in modules._modules.items():
                    ret.append(flatten_model(n))
        except:
            ret.append(modules)
    return flatten_list(ret)
