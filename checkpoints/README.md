# Pretrained checkpoints

This directory is intended for the three original trained checkpoints
associated with the main experimental results reported for CMX, V2.0,
and V2.2.

The checkpoint files are not stored in the GitHub repository because of
their size. Download links will be provided separately.

## Expected directory structure

```text
checkpoints/
├── CMX/
│   └── epoch-300.pth
├── V2_0/
│   └── epoch-300.pth
├── V2_2/
│   └── epoch-300.pth
├── README.md
└── SHA256SUMS
Released checkpoints
ModelCheckpointFull-test mIoU
CMX/CMX/epoch-300.pth83.275%
V2.0V2_0/epoch-300.pth83.768%
V2.2V2_2/epoch-300.pth84.101%


These are the original training-run checkpoints used for the main
CMX–V2.0–V2.2 comparison.
Checkpoints generated during later multi-seed or train/validation
experiments (Phase 7 and Phase 8) are not part of this release.
SHA256 checksums
CMX
7723f37b075774826a26afa7882fcd958b920c5ed5eab81e951ffc107c190158

V2.0
7836a1b4eb2b7461820db19d77e2f9c27ba3730a46a390fe50cf7c34e9766030

V2.2
fa09631877ba16508229b90a3aa5fa7fb2bf20feb27638e6cac26849b216109b

After downloading the checkpoints and placing them in the expected
directories, verify their integrity with:
cd checkpoints
sha256sum -c SHA256SUMS

All three files should report OK.
Evaluation
Each model can automatically locate its released checkpoint.
For CMX:
cd CMX
python eval.py

For V2.0:
cd V2_0
python eval.py

For V2.2:
cd V2_2
python eval.py

A checkpoint located elsewhere can also be evaluated explicitly:
python eval.py --checkpoint /path/to/epoch-300.pth

The default evaluation split is splits/test.txt, containing the
184-image full test set used for the main reported results.
Reproducibility note
The released files are the exact checkpoints associated with the
reported original runs.
The original training runs were not configured for fully deterministic
execution. Consequently, retraining from scratch is not expected to
reproduce the checkpoint files bit-for-bit or necessarily produce
exactly identical mIoU values.
