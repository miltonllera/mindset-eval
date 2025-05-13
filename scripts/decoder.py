import torch.nn as nn


class DecoderWrapper(nn.Module):
    def __init__(self, model, target_dim) -> None:
        super().__init__()

        modules = list(model.children())
        decoders = nn.ModuleDict()

        extracted_features = []
        def hook_fn(module, input, output):
            extracted_features.append(output.detach())

        for i, m in enumerate(modules):
            if isinstance(m,  (nn.Conv2d, nn.Linear)):
                dec = nn.Linear(
                    m.out_features if isinstance(m, nn.Linear)
                    else m.out_channels, target_dim
                )
                m.register_forward_hook(hook_fn)
                decoders[f"Decoder_{i}"] = dec

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
