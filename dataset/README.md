# RGB-T SAR Dataset

This repository contains the experimental splits and dataset documentation
used for the RGB-T semantic segmentation experiments reported with the
CMX, V2.0, and V2.2 models.

The complete dataset is distributed separately because the RGB, thermal,
and semantic-mask files are not stored directly in this GitHub repository.

## Dataset overview

The dataset contains:

- 610 aligned RGB-T image pairs.
- RGB images.
- Thermal infrared (TIR) images.
- Pixel-wise semantic segmentation masks.
- 12 semantic classes.
- An ignore label with value 255.

The experimental partition used for the main results is:

| Split | Images |
|---|---:|
| Training | 426 |
| Test | 184 |
| Total | 610 |

The test set is additionally divided according to acquisition condition:

| Test condition | Images |
|---|---:|
| Day | 103 |
| Tunnel | 63 |
| Night | 18 |
| Full test | 184 |

The main results reported for CMX, V2.0, and V2.2 use the complete
184-image test set.

## Semantic classes

The models predict 12 semantic classes:

| Training ID | Semantic class |
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

The dataset masks originate from the semantic annotation convention:

| Semantic label | Original mask value |
|---|---:|
| Background / Unlabeled | 0 |
| First-Responder | 1 |
| Civilian | 2 |
| Vegetation | 3 |
| Road | 4 |
| Dirt-Road | 5 |
| Building | 6 |
| Undergrowth | 7 |
| Civilian-Car | 8 |
| Responder-Vehicle | 9 |
| Debris | 10 |
| Command-Post | 11 |
| Wall | 12 |
| Ignore | 255 |

During loading, the dataset implementation transforms the semantic mask
representation into the 0-11 class-index representation used by the
network.

The ignore value remains 255 and is excluded from the segmentation loss.

## Expected directory structure

After downloading and extracting the dataset, the recommended structure is:

```text
dataset/
├── rgb/
│   └── ...
├── tir/
│   └── ...
├── masks/
│   └── ...
└── splits/
    ├── train.txt
    ├── test.txt
    ├── test_day.txt
    ├── test_night.txt
    └── test_tunel.txt
```

The RGB, TIR, and mask filenames referenced by the split files must retain
their original names.

Official splits

The following split files are included directly in this GitHub repository:
splits/train.txt
splits/test.txt
splits/test_day.txt
splits/test_night.txt
splits/test_tunel.txt

1. Training split
    - `train.txt`
    - 426 RGB-T pairs.
    - Used for training the models associated with the released checkpoints.

2. Full test split
    - `test.txt`
    - 184 RGB-T pairs.
    - Used to compute the main full-test semantic segmentation results.

3. Day test split
    - `test_day.txt`
    - 103 RGB-T pairs.

4. Tunnel test split
    - `test_tunel.txt`
    - 63 RGB-T pairs.

5. Night test split
    - `test_night.txt`
    - 18 RGB-T pairs.

The three condition-specific test subsets are mutually exclusive and
together constitute the complete 184-image test set.

## Dataset configuration

By default, the code expects the dataset under:
`RGBT_Transformer_Github/dataset/`

A different location can be provided through the environment variable:
`export RGBT_DATASET_ROOT=/path/to/dataset`

The directory referenced by RGBT_DATASET_ROOT must contain:
```
rgb/
tir/
masks/
splits/
```

For example:
```
export RGBT_DATASET_ROOT=/data/RGBT_SAR
cd V2_2
python eval.py
```

## Dataset download
The complete RGB-T dataset will be made available separately.
Download:
[RGB-T Semantic Segmentation UMA-SAR Dataset](https://uma365-my.sharepoint.com/:f:/g/personal/061994342x_uma_es/IgArj3fMeQhdTpArL_1l-TTQAdPmkKi5fTM-pWOneSO7dpU?e=7fibIu)

After downloading it, place or extract the data following the directory
structure described above.

## Reproducibility

The split files included in this repository define the exact experimental
partition used for the main CMX, V2.0, and V2.2 comparison.
The training and test sets are frame-level partitions. Exact frames do not
overlap between training and test sets.
Because multiple frames may originate from the same acquisition scenes,
the evaluation should be interpreted as an in-distribution frame-level
evaluation rather than a scene-disjoint generalization experiment.

## Notes

- RGB image format: PNG.
- TIR image format: PNG.
- Semantic-mask format: PNG.
- Training/evaluation resolution used by the models: 480 x 480.
- Ignore label: 255.
- Number of predicted semantic classes: 12.

## Dataset integrity verification

A SHA256 manifest for the released dataset is included in:
`dataset/SHA256SUMS`
After downloading and extracting the dataset, its integrity can be
verified from the dataset root with:
`sha256sum -c SHA256SUMS`

The manifest covers the RGB images, TIR images, semantic masks, and
official split files used in the experiments.
