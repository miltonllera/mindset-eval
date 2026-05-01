import re


class ActivationRecorder:
    def __init__(self, net, record_from: list[str] | None = None, detach_tensors: bool = True):
        self.net = net
        self.record_from: list[str] = record_from if record_from is not None else ["Conv2d", "Linear"]
        self.detach_tensors = detach_tensors
        self.activation = {}
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for idx, layer in enumerate(self._leaf_layers()):
            name = f"{idx}: {type(layer).__name__}"
            if any(re.search(pattern, name) for pattern in self.record_from):
                self._hooks.append(layer.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(_module, _input, output):
            self.activation[name] = (
                output.detach() if self.detach_tensors else output
            ).cpu()
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

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
