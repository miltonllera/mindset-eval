import re
from typing import Callable, Literal
import torch
import torch.nn as nn
import lightning.pytorch as pl


class FeatureDecoder(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        target_dim: int,
        target_key: str,
        loss: Literal['mse', 'cross_entropy'] | Callable,
        decode_from: list[str] | None = None,
        finetune_model: bool = True
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
        self.net(x)
        decoder_preds = {}
        for k, v in self._extracted_features.items():
            if len(v.shape) > 2:
                v = v.flatten(1)
            decoder_preds[k] = self._decoders[k](v)
        return decoder_preds

    def _step(self, batch, stage):  # type: ignore
        x, y = batch['Image'], batch[self.target_key]

        if len(y.shape) == 1:
            y = y[..., None]
        if isinstance(self.loss, nn.BCEWithLogitsLoss):
            y = y.to(dtype=torch.float32)

        decoder_preds = self.forward(x)

        total_loss, decoder_losses = 0.0, {}
        for k, v in decoder_preds.items():
            decoder_losses[k] = self.loss(v, y)
            total_loss += decoder_losses[k]

        self.log_dict(
            decoder_losses,
            prog_bar=True,
            on_step=stage == 'train',
            on_epoch=stage != 'train',
        )

        return total_loss

    def training_step(self, batch):
        return self._step(batch, 'train')

    def validation_step(self, batch):
        return self._step(batch, 'validation')

    def configure_optimizers(self):
        parameters = [p for p in self._decoders.parameters()]
        if self.finetune_model:
            parameters = [p for p in self.net.parameters()]
        return torch.optim.AdamW(parameters, lr=1e-3)

    def _init_decoders(self):
        input_size = self.net.pretrained_cfg['input_size']  # type: ignore
        sample_input = torch.randn(input_size)[None]  # type: ignore
        self.net.cpu()(sample_input)

        for k, v in self._extracted_features.items():
            if len(v.shape) > 2:
                feature_shape = v.numel()
            else:
                feature_shape = v.shape[-1]

            self._decoders.add_module(k, nn.Linear(feature_shape, self.target_dim))

        self._extracted_features.clear()

    def _register_hooks(self):
        for idx, layer in enumerate(self._leaf_layers()):
            name = f"{idx}: {type(layer).__name__}"
            if any(re.search(pattern, name) for pattern in self.decode_from):
                self._hooks.append(layer.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(_module, _input, output):
            self._extracted_features[name] = (output.detach() if self.finetune_model else output)
        return hook

    def _leaf_layers(self):
        layers = []

        def collect(module):
            for child in module.children():
                if list(child.children()):
                    collect(child)
                else:
                    layers.append(child)

        collect(self.net)
        return layers

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
