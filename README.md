# mindset-eval

**mindset-eval** is a benchmarking and evaluation framework designed to probe deep learning computer vision models on **low- and mid-level vision tasks**. By analyzing intermediate layer representations and linear feature decoders, `mindset-eval` provides insights into how convolutional neural networks (CNNs) and vision transformers (ViTs) process perceptual properties such as amodal completion, visual crowding, shape decomposition, emergent Gestalt features, and 3D depth cues.

---

## 🌟 Key Features

- **Layer Activation Recording**: Intercept and store intermediate activations across `Conv2d` and `Linear` layers using PyTorch forward hooks (`ActivationRecorder`).
- **Linear Feature Probing**: Train linear decoders on top of internal model representations using PyTorch Lightning (`FeatureDecoder`) to evaluate feature readability.
- **Low- & Mid-Level Vision Benchmarks**: Comprehensive evaluation battery including:
  - **Amodal Completion**
  - **Visual Crowding**
  - **Shape Decomposition**
  - **3D & Depth Drawing Perception**
  - **Emergent Gestalt Features**
  - **Non-Accidental Properties (NAP) vs. Metric Properties (MP)**
  - **Relational vs. Coordinate Representation**
- **Seamless `timm` Integration**: Load and benchmark pretrained vision backbones (ResNet, ConvNeXt, DeiT, Swin, etc.) directly from `timm`.
- **Automated Plotting & Metrics**: Calculate similarity metrics (cosine similarity, Euclidean distance, R², accuracy) and output interactive Plotly visualizations.

---

## 📁 Repository Structure

```text
mindset-eval/
├── bin/                       # Executable shell scripts for batch benchmarking
│   └── low_mid_vis/
│       └── amodal_completion.sh
├── data/                      # Local datasets, models, and output results
│   ├── datasets/
│   ├── models/
│   └── results/
├── notebooks/                 # Jupyter notebooks for interactive analysis
│   └── model_tests.ipynb
├── scripts/                   # Evaluation scripts and utilities
│   ├── download.py            # Dataset downloader (via Kaggle)
│   └── low_mid_vis/           # Low- and mid-level vision task scripts
│       ├── amodal_completion.py
│       ├── crowding.py
│       ├── decomposition.py
│       ├── depth_drawings.py
│       ├── emergent_features.py
│       ├── nap_vs_mp.py
│       └── rel_vs_coord.py
├── src/                       # Core python package
│   ├── activation_recorder.py # PyTorch hook module for layer activations
│   ├── dataset.py             # Polars-backed PyTorch AnnotatedDataset
│   ├── feature_decoder.py     # PyTorch Lightning feature decoder
│   └── utils.py               # Model initialization & helper functions
└── pyproject.toml             # Project metadata & dependencies (uv compatible)
```

---

## ⚙️ Installation

### Prerequisites

- Python `>= 3.13`
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- Chrome or Chromium, needed by `kaleido>=1.2` to save plots with `write_image`

### Quick Start with `uv`

1. Clone the repository:
   ```bash
   git clone https://github.com/mindset-vision/mindset-eval.git
   cd mindset-eval
   ```

2. Synchronize dependencies:
   ```bash
   uv sync
   ```

---

## 📊 Downloading Datasets

Use the included dataset downloader script to fetch the evaluation datasets:

```bash
# Download the lite dataset version (default)
uv run python -m scripts.download --version lite --path data/datasets/mindset-lite

# Download the full dataset version
uv run python -m scripts.download --version full --path data/datasets/mindset
```

---

## 🧪 Evaluation Benchmarks

The repository includes specialized scripts under `scripts/low_mid_vis/` for evaluating various perceptual properties:

| Script | Vision Task / Probing Target | Output Metric / Analysis |
| :--- | :--- | :--- |
| `amodal_completion.py` | Control vs. Occluded vs. Notched shape representations | Layer-wise Cosine Similarity / Distance |
| `crowding.py` | Vernier offset decoding under visual crowding conditions | Linear Probing Accuracy via PyTorch Lightning |
| `decomposition.py` | Unified vs. Naturally & Unnaturally split shape representations | Layer-wise Cosine Similarity |
| `depth_drawings.py` | Representation invariance across 2D/3D rotated shapes & hulls | Layer-wise Distance Comparison |
| `emergent_features.py` | Probing emergent Gestalt features vs. individual components | Representation Distance Analysis |
| `nap_vs_mp.py` | Sensitivity to Non-Accidental (NAP) vs. Metric Properties (MP) | Relative Feature Distances |
| `rel_vs_coord.py` | Sensitivity to relational changes vs. coordinate shifts | Layer-wise Feature Shifts |

---

## 🚀 Usage Examples

### 1. Running Benchmark Scripts via CLI

You can run individual benchmark evaluations from the terminal:

```bash
# Evaluate Amodal Completion on ConvNeXt and DeiT models
uv run python -m scripts.low_mid_vis.amodal_completion \
  --annotations "data/datasets/full/low_mid_level_vision/amodal_completion/annotation.csv" \
  --models convnext_tiny_imagenet_full_seed-0 deit_base_imagenet_full_seed-0 \
  --results_folder data/results

# Run Visual Crowding Linear Feature Decoder
uv run python -m scripts.low_mid_vis.crowding \
  --annotations "data/datasets/full/low_mid_level_vision/crowding/annotation.csv" \
  --models resnet50 \
  --results_folder data/results
```

Alternatively, use the shell scripts located in `bin/low_mid_vis/`:
```bash
bash bin/low_mid_vis/amodal_completion.sh
```

---

### 2. V-JEPA 2.1 and the large-model runs

`init_model()` also supports `vjepa2_1_vit_base_384` (12 blocks) and
`vjepa2_1_vit_large_384` (24 blocks). These are distilled RGB feature encoders,
not classifiers: input is BCHW at 384 px and output is B×576×768/1024.
The usual task CLIs and `blocks.N` hooks work without a separate launcher.

On first use, the loader downloads the [Meta source at revision
204698b](https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812)
and the official [B weights](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt)
or [L weights](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt)
to `data/models/vjepa2/`. Weight downloads are SHA-256 checked against the checkpoints
used for our earlier runs; cached files are reused without downloading again.
`pretrained=False` fetches only the source. To reuse an offline cache, set
`MINDSET_VJEPA_MODEL_DIR` to a directory containing the `.pt` files and the
`vjepa2-204698b45b3712590f06245fbfba32d3be539812/` source tree. Meta's source and
weights retain their upstream licenses.

The four model×task combinations below are the held SigLIP2 and V-JEPA-L runs.
Each command runs both `pre_act` and `res_stream`, so there are **eight runs**
in total. Run them sequentially on a machine with enough host RAM and VRAM;
these are not laptop smoke tests. Crowding trains a decoder at every selected
layer, and decomposition retains full token activations. Keep the default
resolution, precision, batch size and decoder settings when comparing results.

Run `uv sync --locked` first, then paste this helper into a Bash shell from the
repository root. It keeps the two stream settings in one place:

```bash
run_both_streams() {
  local task="$1" model="$2" dataset="$3" results="$4" residual="$5"
  local tag record
  for tag in pre_act res_stream; do
    record='^blocks\.([0-9]|[1-9][0-9]):out'
    if [ "$tag" = res_stream ]; then record="$residual"; fi
    uv run --locked python -m "scripts.low_mid_vis.$task" \
      --annotations_file "data/datasets/low_mid_vision/$dataset/annotation.csv" \
      --models "$model" --record_from "$record" --overwrite_recordings \
      --results_folder "$results" --output_tag "$tag" || return
  done
}
```

**SigLIP2 — crowding**

```bash
run_both_streams crowding vit_so400m_patch14_siglip_378.v2_webli \
  uncrowding_distributions_large_offset experiments/exp_006_siglip2/.intermediates/eval \
  '^blocks\.([0-9]|[1-9][0-9])\.drop_path2'
```

**SigLIP2 — decomposition**

```bash
run_both_streams decomposition vit_so400m_patch14_siglip_378.v2_webli \
  decomposition experiments/exp_006_siglip2/.intermediates/eval \
  '^blocks\.([0-9]|[1-9][0-9])\.drop_path2'
```

**V-JEPA-L — crowding**

```bash
run_both_streams crowding vjepa2_1_vit_large_384 \
  uncrowding_distributions_large_offset experiments/exp_009_vjepa21_l/.intermediates/eval \
  '^blocks\.[0-9]+\.drop_path'
```

**V-JEPA-L — decomposition**

```bash
run_both_streams decomposition vjepa2_1_vit_large_384 \
  decomposition experiments/exp_009_vjepa21_l/.intermediates/eval \
  '^blocks\.[0-9]+\.drop_path'
```

The annotation paths assume the same toolbox-generated datasets, including the
large-offset crowding distribution, not the similarly named lite datasets.
Crowding arguments come from the `exp_006_siglip2` / `exp_009_vjepa21_l`
`plan-crowding.json` manifests. Decomposition arguments follow the completed
`exp_007_dinov3_l/.results/manifests/task-14.json` and `task-15.json`, with the
model, output directory and model-specific stream selector replaced. The held
decomposition jobs have not been run. Outputs go under `uncrowding/<model>/`
or `decomposition/<model>/`, with the stream tag in each parquet/PNG filename.

---

### 3. Python API Examples

#### Recording Layer Activations (`ActivationRecorder`)

```python
import torch
from src.utils import init_model
from src.activation_recorder import ActivationRecorder

# 1. Initialize a pretrained model from timm
model = init_model("resnet50")

# 2. Attach ActivationRecorder to capture Conv2d and Linear layers
recorder = ActivationRecorder(model, record_from=["Conv2d", "Linear"])

# 3. Perform forward pass
dummy_input = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)
model(dummy_input)

# 4. Access recorded layer activations
for layer_name, tensor in recorder.activation.items():
    print(f"Layer: {layer_name} | Activation Shape: {tensor.shape}")

# 5. Clean up hooks when finished
recorder.remove_hooks()
```

#### Training Linear Feature Decoders (`FeatureDecoder`)

```python
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from src.utils import init_model, model_transform
from src.dataset import AnnotatedDataset
from src.feature_decoder import FeatureDecoder

# 1. Initialize network and transform
net = init_model("resnet50")
transform = model_transform(net)

# 2. Load dataset
dataset = AnnotatedDataset(
    annotations_file="data/datasets/full/low_mid_level_vision/crowding/annotation.csv",
    test_columns=["Path", "VernierType"],
    transform=transform,
)
dataloader = DataLoader(dataset, batch_size=32)

# 3. Create PyTorch Lightning FeatureDecoder for target variable
decoder = FeatureDecoder(
    net=net,
    target_dim=1,
    target_key="VernierType",
    loss="cross_entropy",
    finetune_model=False,
)

# 4. Train decoder using Lightning Trainer
trainer = pl.Trainer(max_epochs=5)
trainer.fit(decoder, dataloader)
```

#### Loading Annotated Datasets (`AnnotatedDataset`)

```python
from src.dataset import AnnotatedDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = AnnotatedDataset(
    annotations_file="data/datasets/full/low_mid_level_vision/amodal_completion/annotation.csv",
    test_columns=["SampleID", "ControlPath", "OccludedPath"],
    transform=transform,
)

print(f"Dataset length: {len(dataset)}")
sample = dataset[0]
print("Sample keys:", sample.keys())
# Output contains 'SampleID', 'ControlPath', 'ControlImage', 'OccludedPath', 'OccludedImage'
```

---

## 📜 License

This project is open source. Refer to repository header or contact [mindset-vision](https://github.com/mindset-vision) for dataset licensing details.
