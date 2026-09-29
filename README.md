# RGB-T Transformer Semantic Segmentation for SAR Environments

Research implementation and reproducibility package for RGB-T semantic
segmentation in search-and-rescue (SAR) environments.

This repository contains three Transformer-based RGB-T semantic
segmentation implementations:

- **CMX** — baseline derived from the official CMX implementation.
- **V2.0** — research variant introducing a hybrid directional/global
  attention mechanism.
- **V2.2** — refinement of V2.0 that applies the projection and dropout
  stages after hybrid-feature concatenation.

The repository also contains the experimental dataset splits, Docker
environment, training/evaluation scripts, checkpoint integrity hashes,
and documentation required to reproduce the main experiments.

---

## Repository structure

```text
RGBT_Transformer_Github/
├── CMX/
├── V2_0/
├── V2_2/
│
├── checkpoints/
│   ├── README.md
│   └── SHA256SUMS
│
├── dataset/
│   ├── README.md
│   └── splits/
│       ├── train.txt
│       ├── test.txt
│       ├── test_day.txt
│       ├── test_night.txt
│       └── test_tunel.txt
│
├── pretrained/
│   ├── README.md
│   └── segformer/
│
├── docker/
│   ├── Dockerfile
│   └── README.md
│
├── scripts/
│   ├── train_cmx.sh
│   ├── train_v2_0.sh
│   ├── train_v2_2.sh
│   ├── eval_cmx.sh
│   ├── eval_v2_0.sh
│   └── eval_v2_2.sh
│
├── requirements.txt
├── LICENSE
├── NOTICE.md
├── .gitignore
└── .dockerignore
```

---

## Model variants

### CMX

`CMX/` contains the baseline implementation derived from the official
CMX RGB-X semantic segmentation code.

CMX serves as the architectural reference for the experimental
comparison.

### V2.0

`V2_0/` introduces a hybrid attention design in the first encoder stage.

For the Stage-1 feature representation, the feature channels are divided
between directional and global-attention branches. The directional branch
models horizontal and vertical dependencies, while the global branch
retains spatially reduced global attention.

The two branches are subsequently concatenated.

### V2.2

`V2_2/` retains the V2.0 hybrid-attention structure but applies the
projection and projection-dropout operations after concatenating the
directional and global representations.

The three implementations are intentionally kept as separate source trees
so that the exact experimental models can be inspected independently.

---

## Dataset

The RGB-T SAR dataset used in the main experiments contains:

| Partition | RGB-T pairs |
|---|---:|
| Training | 426 |
| Test | 184 |
| **Total** | **610** |

The complete test partition is additionally divided by acquisition
condition:

| Condition | Images |
|---|---:|
| Day | 103 |
| Tunnel | 63 |
| Night | 18 |
| **Full test** | **184** |

The repository contains the exact split files used in the experiments.

The RGB images, TIR images, and segmentation masks are distributed
separately and are intentionally not committed to Git.

See `dataset/README.md`.

### Dataset layout

The default expected layout is:

```text
dataset/
├── rgb/
├── tir/
├── masks/
└── splits/
    ├── train.txt
    ├── test.txt
    ├── test_day.txt
    ├── test_night.txt
    └── test_tunel.txt
```

An external dataset location can be selected with:

```bash
export RGBT_DATASET_ROOT=/path/to/dataset
```

### Evaluation protocol

The principal comparison uses the 184-image `test.txt` partition.

The Day, Tunnel, and Night subsets are provided for condition-specific
analysis.

The train/test partition is frame-level rather than scene-disjoint.
Exact frames do not overlap between training and testing, although frames
from the same acquisition scenes may occur in both partitions.

Accordingly, these results should be interpreted as an in-distribution
frame-level evaluation and not as a scene-disjoint generalization test.

---

## Semantic classes

The network predicts 12 semantic classes:

| Network ID | Class |
|---:|---|
| 0 | First-Responder |
| 1 | Civilian |
| 2 | Vegetation |
| 3 | Road |
| 4 | Dirt-Road |
| 5 | Building |
| 6 | Undergrowth |
| 7 | Civilian-Car |
| 8 | Responder-Vehicle |
| 9 | Debris |
| 10 | Command-Post |
| 11 | Wall |

The source annotation convention uses semantic values 1-12 for these
classes and 255 as the ignore label. The dataset loader converts the
trainable class representation to indices 0-11.

---

## Original experimental results

The following values correspond to the original training runs associated
with the released `epoch-300.pth` checkpoints.

| Model | Full mIoU | Day mIoU | Night mIoU | Tunnel mIoU |
|---|---:|---:|---:|---:|
| CMX | 83.275% | 81.640% | 72.719% | 75.853% |
| V2.0 | 83.768% | 82.588% | 75.551% | 75.792% |
| V2.2 | 84.101% | 82.961% | 74.980% | 75.736% |

These values describe the original training runs.

### Model complexity

The three implementations contain approximately the same number of
parameters:

| Model | Parameters | GFLOPs |
|---|---:|---:|
| CMX | 181.0605 M | 143.504554578 |
| V2.0 | 181.0605 M | 142.090791444 |
| V2.2 | 181.0605 M | 142.444685838 |

The complexity measurements correspond to the common experimental input
configuration used by the project.

---

## Experimental environment

The experiments were performed with:

```text
Python                    3.11.10
PyTorch                   2.5.1+cu124
torchvision               0.20.1+cu124
CUDA                      12.4.1
NumPy                     1.26.4
SciPy                     1.13.1
timm                      0.9.16
OpenCV                    4.10.0.84
fvcore                    0.1.5.post20221221
```

The complete dependency set is pinned in `requirements.txt`.

A reproducible container definition is provided in:

```text
docker/Dockerfile
```

---

## Docker installation

From the repository root:

```bash
docker build \
    -f docker/Dockerfile \
    -t rgbt-transformer:cu124 \
    .
```

Run interactively with GPU support:

```bash
docker run --rm -it \
    --gpus all \
    --ipc=host \
    -v "$PWD":/workspace/RGBT_Transformer_Github \
    -w /workspace/RGBT_Transformer_Github \
    rgbt-transformer:cu124
```

Verify the environment:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

See `docker/README.md` for additional information.

---

## Pretrained MiT-B5 backbone

CMX, V2.0, and V2.2 use the same pretrained SegFormer MiT-B5 backbone.

Expected location:

```text
pretrained/segformer/mit_b5.pth
```

Expected SHA256:

```text
d5c381730c305a9b9d411c2ec9768fb40e922298ecdb4bfe0a606ccd12af0225
```

An external path can alternatively be specified using:

```bash
export RGBT_PRETRAINED=/path/to/mit_b5.pth
```

See `pretrained/README.md`.

---

## Released checkpoints

The trained checkpoints are not committed to GitHub because each file is
approximately 2 GB. Therefore, you can downloaded from here: [CMX](https://uma365-my.sharepoint.com/:f:/g/personal/061994342x_uma_es/IgB_DmDNLJdXTIu5_6S2btUjAad7lQ0kBkuQlDR9EbwdGZA?e=F4qT1p), [V2.2](https://uma365-my.sharepoint.com/:f:/g/personal/061994342x_uma_es/IgBdXuC8p4OkSawX1yqpMCFDAf7Q-hsTNjZyVv8-DHVOFMs?e=oCGlFJ), [V2.0](https://uma365-my.sharepoint.com/:f:/g/personal/061994342x_uma_es/IgACrjezNw6qRpcX8KKoMamxASb4nlZTc5GO72ueUPYQyk4?e=JNrxIZ)

Expected structure:

```text
checkpoints/
├── CMX/
│   └── epoch-300.pth
├── V2_0/
│   └── epoch-300.pth
└── V2_2/
    └── epoch-300.pth
```

Integrity hashes:

| Model | SHA256 |
|---|---|
| CMX | `7723f37b075774826a26afa7882fcd958b920c5ed5eab81e951ffc107c190158` |
| V2.0 | `7836a1b4eb2b7461820db19d77e2f9c27ba3730a46a390fe50cf7c34e9766030` |
| V2.2 | `fa09631877ba16508229b90a3aa5fa7fb2bf20feb27638e6cac26849b216109b` |

The same values are stored in `checkpoints/SHA256SUMS`.

After downloading the files:

```bash
cd checkpoints
sha256sum -c SHA256SUMS
```

Dataset and checkpoint download locations will be added to the release
documentation when the external files are made publicly available.

---

## Training

Training launchers are provided under `scripts/`.

The original released runs used two DDP processes.

### CMX

```bash
./scripts/train_cmx.sh
```

### V2.0

```bash
./scripts/train_v2_0.sh
```

### V2.2

```bash
./scripts/train_v2_2.sh
```

The number of processes can be changed if required:

```bash
NPROC_PER_NODE=4 ./scripts/train_v2_2.sh
```

A dataset outside the repository can be used with:

```bash
RGBT_DATASET_ROOT=/data/RGBT_SAR \
./scripts/train_v2_2.sh
```

Training outputs are written under:

```text
outputs/
├── CMX/
├── V2_0/
└── V2_2/
```

The `outputs/` directory is excluded from Git.

---

## Evaluation

Once the dataset and released checkpoints are available in their default
locations, evaluation can be executed directly from the repository root.

### CMX

```bash
./scripts/eval_cmx.sh
```

### V2.0

```bash
./scripts/eval_v2_0.sh
```

### V2.2

```bash
./scripts/eval_v2_2.sh
```

A checkpoint in another location can be selected explicitly:

```bash
./scripts/eval_v2_2.sh \
    --checkpoint /path/to/epoch-300.pth
```

A GPU can be selected with:

```bash
./scripts/eval_v2_2.sh -d 0
```

Predictions can be saved using the evaluator's `--save_path` argument.

---

## Upstream CMX attribution

The CMX baseline is derived from the official implementation of:

**CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation with
Transformers**

Jiaming Zhang, Huayao Liu, Kailun Yang, Xinxin Hu, Ruiping Liu, and
Rainer Stiefelhagen.

IEEE Transactions on Intelligent Transportation Systems, 2023.

Upstream source repository:

```text
huaaaliu/RGBX_Semantic_Segmentation
```

The upstream implementation is distributed under the MIT License.

The original license is preserved in `LICENSE`, and attribution details
are provided in `NOTICE.md`.

---

## License

See `LICENSE` and `NOTICE.md`.

The retained CMX-derived source remains subject to the original MIT
license and attribution requirements.

---

## Citation

Citation information for the associated RGB-T SAR research article will
be added when the publication metadata is available.

When using the CMX-derived implementation, please also cite the original
CMX publication described in the attribution section above.
