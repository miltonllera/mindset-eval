import sys
from pathlib import Path
import unittest
import torch
import timm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.layer_spec import (
    parse_layer_spec,
    resolve_layer_targets,
    extract_stream_tensor,
    sanitize_key_for_module_dict,
)
from src.activation_recorder import ActivationRecorder
from src.feature_decoder import FeatureDecoder


class TestLayerSpec(unittest.TestCase):
    def test_parse_layer_spec(self):
        self.assertEqual(parse_layer_spec("stem:out"), ("stem", ["out"], True))
        self.assertEqual(parse_layer_spec("stem:in"), ("stem", ["in"], True))
        self.assertEqual(parse_layer_spec("stem:delta"), ("stem", ["delta"], True))
        self.assertEqual(parse_layer_spec("stem:all"), ("stem", ["in", "out", "delta"], True))
        self.assertEqual(parse_layer_spec("stem"), ("stem", ["out"], False))
        self.assertEqual(parse_layer_spec("0: Conv2d"), ("0: Conv2d", ["out"], False))

    def test_sanitize_key(self):
        self.assertEqual(
            sanitize_key_for_module_dict("stages.0.blocks.0:out"),
            "stages__0__blocks__0:out",
        )

    def test_activation_recorder_streams(self):
        model = timm.create_model("convnext_tiny", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        rec = ActivationRecorder(
            model,
            record_from=["stem:out", r"stages\.0\.blocks\.0:all"],
        )
        model(x)

        expected_keys = [
            "stem:out",
            "stages.0.blocks.0:in",
            "stages.0.blocks.0:out",
            "stages.0.blocks.0:delta",
        ]
        for k in expected_keys:
            self.assertIn(k, rec.activation)
            self.assertEqual(rec.activation[k].shape[0], 2)

        # Verify delta == out - in
        in_t = rec.activation["stages.0.blocks.0:in"]
        out_t = rec.activation["stages.0.blocks.0:out"]
        delta_t = rec.activation["stages.0.blocks.0:delta"]
        torch.testing.assert_close(delta_t, out_t - in_t)
        rec.remove_hooks()

    def test_feature_decoder_preds(self):
        model = timm.create_model("convnext_tiny", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        decoder = FeatureDecoder(
            model,
            target_dim=1,
            target_key="Target",
            loss="cross_entropy",
            decode_from=["stem:out", r"stages\.0\.blocks\.0:delta"],
        )
        preds = decoder(x)
        self.assertIn("stem:out", preds)
        self.assertIn("stages.0.blocks.0:delta", preds)
        self.assertEqual(preds["stem:out"].shape, torch.Size([2, 1]))
        self.assertEqual(preds["stages.0.blocks.0:delta"].shape, torch.Size([2, 1]))

    def test_legacy_support(self):
        model = timm.create_model("convnext_tiny", pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        rec = ActivationRecorder(model, record_from=["^0: Conv2d$"])
        model(x)
        self.assertIn("0: Conv2d", rec.activation)
        rec.remove_hooks()


if __name__ == "__main__":
    unittest.main()
