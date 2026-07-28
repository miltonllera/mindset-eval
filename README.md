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
│       ├── nap_vs_mp_change.py
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
| `nap_vs_mp_change.py` | Feature changes across NAP and MP alterations | Comparative Layer-wise Analysis |
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

### 2. Python API Examples

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
