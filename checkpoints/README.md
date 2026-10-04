# Pretrained checkpoints

This directory is intended for the three original trained checkpoints
associated with the main experimental results reported for CMX, V2.0,
and V2.2.

The checkpoint files are not stored in the GitHub repository because of
their size. Download links will be provided separately.

## Expected directory structure

```
checkpoints/
├── CMX/
│   └── epoch-300.pth
├── V2_0/
│   └── epoch-300.pth
├── V2_2/
│   ├── mit_bo
|   |       └──epoch-300.pth
|   ├── mit_b1
|   |       └──epoch-300.pth
|   ├── mit_b2
|   |       └──epoch-300.pth
|   ├── mit_b3
|   |       └──epoch-300.pth
|   ├── mit_b4
|   |       └──epoch-300.pth
|   ├── mit_b5
|   |       └──epoch-300.pth
├── README.md
└── SHA256SUMS
```

## Released checkpoints

|ModelCheckpoint|   Full-test                   |   mIoU    |
|---------------|:-----------------------------:|:---------:|  
|CMX            |   CMX/epoch-300.pth           |   83.275% |
|V2.0           |   V2_0/epoch-300.pth          |   83.768% |
|V2.2_mit_b5    |   V2_2/mit_b5/epoch-300.pth   |   84.101% |
|V2.2_mit_b4    |   V2_2/mit_b4/epoch-300.pth   |   --.---% |
|V2.2_mit_b3    |   V2_2/mit_b3/epoch-300.pth   |   --.---% |
|V2.2_mit_b2    |   V2_2/mit_b2/epoch-300.pth   |   --.---% |
|V2.2_mit_b1    |   V2_2/mit_b1/epoch-300.pth   |   83.530% |
|V2.2_mit_b0    |   V2_2/mit_b0/epoch-300.pth   |   82.300% |


These are the original training-run checkpoints used for the main
CMX–V2.0–V2.2 comparison.

## SHA256 checksums

- CMX
    - 7723f37b075774826a26afa7882fcd958b920c5ed5eab81e951ffc107c190158

- V2.0
    - 7836a1b4eb2b7461820db19d77e2f9c27ba3730a46a390fe50cf7c34e9766030

- V2.2/mit_b5
    - fa09631877ba16508229b90a3aa5fa7fb2bf20feb27638e6cac26849b216109b

- V2.2/mit_b4
    - xxx

- V2.2/mit_b3
    - xxx

- V2.2/mit_b2
    - xxx

- V2.2/mit_b1
    - cf7e48d1ddbfd63c1fd07dbae732272cce9d13667d6207e497f01438dca583e0

- V2.2/mit_b0
    - d3694fa4b6ca15a3de96a6469d20107ae0e7f2a7468aaa1751962bdc36ad7ded

After downloading the checkpoints and placing them in the expected
directories, verify their integrity with:
```
cd checkpoints
sha256sum -c SHA256SUMS
```

All three files should report OK.
Evaluation
Each model can automatically locate its released checkpoint.

__For CMX:__
```
cd CMX
python eval.py
```

__For V2.0:__
```
cd V2_0
python eval.py
```

__For V2.2:__

```
cd V2_2
python eval.py
```

A checkpoint located elsewhere can also be evaluated explicitly:
`python eval.py --checkpoint /path/to/epoch-300.pth`

The default evaluation split is `splits/test.txt`, containing the
184-image full test set used for the main reported results.
