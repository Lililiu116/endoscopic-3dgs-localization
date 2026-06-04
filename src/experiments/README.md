# Experiments

This folder contains the main scripts used in the endoscopic 3DGS localization pipeline.

All commands are expected to be run from the `src/` directory.

## Main Workflow

The typical preprocessing and training workflow is:

1. Split the ShapeSplat dataset.
2. Extract 2D image features with XFeat.
3. Generate offline ground-truth 2D–3D correspondences.
4. Train the matching model.

## Scripts

### `04_split_dataset.py`

Generates train / validation / test splits for the ShapeSplat dataset.

Example:

    python -u -m experiments.04_split_dataset

### `03_XFeat_inference.py`

Extracts 2D image keypoints and descriptors using XFeat.

Example:

    python -u -m experiments.03_XFeat_inference

### `07_gt_corresponding.py`

Generates offline ground-truth 2D–3D correspondences between XFeat image keypoints and downsampled 3D Gaussian primitives.

The correspondence generation uses rendered depth / weight information and nearest-neighbor filtering to associate 2D keypoints with valid Gaussian primitives.

Example:

    python -u -m experiments.07_gt_corresponding

### `08_train.py`

Main training script for the 2D–3D matching model.

Example:

    python -u -m experiments.08_train --name best

The `--name` argument selects an experiment configuration from:

    config/config_shapesplat.yaml

Common configurations include:

- `best`: final ShapeSplat configuration
- `scared`: fine-tuning on SCARED with the proposed matcher
- `gomatch`: GoMatch baseline on ShapeSplat
- `scaredgomatch`: GoMatch baseline / fine-tuning on SCARED

## Notes

These scripts depend on prepared datasets, pretrained weights, external repositories, and local path configuration. Please update the corresponding configuration files before running them.