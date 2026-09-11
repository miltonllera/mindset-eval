import logging
import torch.nn as nn
from src.layer_spec import resolve_layer_targets, extract_stream_tensor

_logger = logging.getLogger(__name__)


class ActivationRecorder:
    def __init__(self, net: nn.Module, record_from: list[str] | None = None, detach_tensors: bool = True):
        self.net = net
        self.record_from: list[str] = record_from if record_from is not None else ["Conv2d", "Linear"]
        self.detach_tensors = detach_tensors
        self.activation = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        targets = resolve_layer_targets(self.net, self.record_from)
        for name, layer, stream in targets:
            self._hooks.append(layer.register_forward_hook(self._make_hook(name, stream)))

    def _make_hook(self, name: str, stream: str):
        def hook(_module, args, output):
            val = extract_stream_tensor(args, output, stream, layer_name=name)
            self.activation[name] = (
                val.detach() if self.detach_tensors else val
            ).cpu()
        return hook

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
