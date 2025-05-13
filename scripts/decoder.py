import torch
import torch.nn as nn
import torchvision.ops as vnn


class DecoderWrapper(nn.Module):
    def __init__(self, model: nn.Module, target_dim: int) -> None:
        super().__init__()

        modules = flatten_model(list(model.modules()))
        decoders = nn.ModuleDict()

        input_size = model.pretrained_cfg['input_size']  # type: ignore
        sample_input = torch.randn(input_size)[None]  # type: ignore

        extracted_features = []
        def hook_fn(module, input, output):
            extracted_features.append(output.detach())

        for i, m in enumerate(modules):
            if isinstance(m,  (nn.Conv2d, nn.Linear)):
                m.register_forward_hook(hook_fn)
        model(sample_input)

        for i, features in enumerate(extracted_features):
            input_dims = features.numel()
            dec = nn.Sequential(
                nn.Flatten(1, -1),
                nn.Linear(input_dims, target_dim)
            )
            decoders[f"Decoder_{i}"] = dec

        extracted_features.clear()

        for p in model.parameters():
            p.requires_grad = False

        self.model = model
        self.decoders = decoders
        self.extracted_features = extracted_features

    def forward(self, x):
        self.model(x)
        decoder_preds = []
        for features, dec in zip(self.extracted_features, self.decoders.values()):
            decoder_preds.append(dec(features))
        self.extracted_features.clear()
        return decoder_preds

    def clear_features(self):
        self.extracted_features.clear()


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
