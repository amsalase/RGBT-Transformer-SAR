# Docker environment

The experiments were executed using a CUDA 12.4 / PyTorch 2.5.1
environment.

## Original experimental environment

The locally archived experimental Docker image was:

```text
amsaladopytorch:rgbx-cu124

Recorded image identifier:
sha256:f51058803f388128006acffe5da7b70160f764ea58d858a608b84d1e2fadf5ad

Image creation timestamp:
2026-09-05T17:04:28.453897951+02:00

The important runtime versions were:
Python                    3.11.10
PyTorch                   2.5.1+cu124
torchvision               0.20.1+cu124
CUDA                      12.4.1
numpy                     1.26.4
scipy                     1.13.1
easydict                  1.13
opencv-python-headless    4.10.0.84
timm                      0.9.16
tqdm                      4.66.5
tensorboardX              2.6.2.2
tensorboard               2.17.1
matplotlib                3.9.2
pandas                    2.2.2
scikit-learn              1.5.2
Pillow                    10.4.0
PyYAML                    6.0.2
fvcore                    0.1.5.post20221221

Build
Run from the repository root:
docker build \
    -f docker/Dockerfile \
    -t rgbt-transformer:cu124 \
    .

Interactive container
From the repository root:
docker run --rm -it \
    --gpus all \
    --ipc=host \
    -v "$PWD":/workspace/RGBT_Transformer_Github \
    -w /workspace/RGBT_Transformer_Github \
    rgbt-transformer:cu124

Verify CUDA
Inside the container:
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"

For a GPU-enabled host, the expected environment should report PyTorch
2.5.1+cu124 and CUDA 12.4 support.
Multi-GPU training
The model implementations use PyTorch Distributed Data Parallel (DDP).
For example, from the repository root inside the container:
torchrun \
    --standalone \
    --nproc_per_node=2 \
    CMX/train.py

Equivalent commands can be used for:
V2_0/train.py
V2_2/train.py

The released original checkpoints were generated from the corresponding
model implementations and the documented experimental dataset split.
