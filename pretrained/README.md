# Pretrained backbone

CMX, V2.0, and V2.2 use the same pretrained SegFormer MiT-B5 backbone.

The pretrained weight file is not stored in this GitHub repository.

## Expected location

Place the file at:

```
pretrained/
└── segformer/
    └── mit_b5.pth
```

By default, all three model configurations look for:
`RGBT_Transformer_Github/pretrained/segformer/mit_b5.pth`

An alternative location can be specified with:
`export RGBT_PRETRAINED=/path/to/mit_b5.pth`

__Expected filename:__
`mit_b5.pth`

__SHA256:__
d5c381730c305a9b9d411c2ec9768fb40e922298ecdb4bfe0a606ccd12af0225

The same pretrained file was used by CMX, V2.0, and V2.2.
Verify a downloaded file with:
`sha256sum mit_b5.pth`

The resulting SHA256 must match the value above.

## Download
The pretrained MiT-B5 weight must be obtained separately.
A verified download source will be documented in the main repository
README.

