# VID-Trans-ReID Camera-Removed Only (Cleaned)

This cleaned version is for the **first ablation only**:

- remove camera metadata from the transformer backbone
- do **not** add MetaBIN
- do **not** add GRL

## What was cleaned

- removed camera embedding usage from `vit_ID.py`
- removed `cam_label` dependency from model forward path
- fixed missing `--model_path` argument in training
- added `--output_dir`, `--epochs`, and `--eval_every`
- cleaned training and test scripts for the camera-removed-only baseline
- fixed center-loss optimization so both global and local center criteria are stepped

## Train

```bash
python VID_Trans_ReID.py \
  --Dataset_name Mars \
  --model_path /path/to/jx_vit_base_p16_224-80ecf9dd.pth \
  --output_dir ./output_camera_removed \
  --epochs 120 \
  --eval_every 10
```

## Test

```bash
python VID_Test.py \
  --Dataset_name Mars \
  --model_path ./output_camera_removed/Mars_camera_removed_best.pth
```

## Important note

This repo is intended to measure the performance drop caused by **removing camera metadata only**. Since the original VID-Trans-ReID uses camera embedding as a strong signal, a noticeable Rank-1 and mAP drop is expected.
