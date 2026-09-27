#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

echo "============================================================"
echo "Training V2.0"
echo "Repository       : ${REPO_ROOT}"
echo "Processes / node : ${NPROC_PER_NODE}"
echo "Dataset          : ${RGBT_DATASET_ROOT:-${REPO_ROOT}/dataset}"
echo "Pretrained       : ${RGBT_PRETRAINED:-${REPO_ROOT}/pretrained/segformer/mit_b5.pth}"
echo "============================================================"

cd "${REPO_ROOT}"

torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    V2_0/train.py \
    "$@"
