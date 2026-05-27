# Multitask Learning for Geometric Shape Classification and Counting

A multitask CNN that simultaneously classifies pairs of geometric shapes and regresses their counts from 28×28 grayscale images.

## Problem

The dataset contains 9,000 grayscale images, each with two types of shapes (from 6 possible: squares, circles, and four triangle orientations) whose counts always sum to 10. This produces 135 unique classes (15 shape pairs × 9 count combinations). The model must solve two tasks at once: identify the class and predict the count of each shape type.

## Architecture

- **Backbone:** 4-layer CNN (1→8→16→32→64 channels, 3×3 kernels) followed by a dense projection to 256 features.
- **Classification head:** Dropout (p=0.7) → Linear(256→135) → LogSoftmax
- **Regression head:** Linear(256→6), predicting counts for each shape type.
- **Loss:** Weighted combination of NLL (classification) and MSE (regression), controlled by λ_cls and λ_cnt.

## Data Augmentation

Rotations (90°, 180°, −90°) and flips with corresponding permutations of the ground truth count vector — e.g., a 90° rotation shifts the "Up" triangle count to the "Right" index. This was critical for combating overfitting on the classification task.

## Results

| Experiment | Configuration | Accuracy | RMSE |
|---|---|---|---|
| 1 | Classification only | 51.80% | — |
| 2 | Regression only | — | 0.5853 |
| 3 | Multitask (λ=1.0/1.0) | 50.50% | 0.6156 |

Key finding: the regression task converges very quickly (~20 epochs, loss ≈ 0.1) while classification remains harder (loss ≈ 1.0 with heavy dropout). The count task neither helps nor hurts classification performance in the multitask setup.

## Tech Stack

Python, PyTorch, Google Colab (GPU)

## Files

- `multitask.ipynb` — Full training pipeline, experiments, and evaluation
- `Report.pdf` — [Detailed analysis and discussion](./Report.pdf)
