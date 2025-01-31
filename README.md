# RefineThenCalibrate-Vision

Code to run the computer vision benchmark in the paper "Rethinking Early Stopping: Refine, Then Calibrate".

## Files
- `main.py`: Launch runs and log results.
- `utils.py`: Contains our pytorch-lightning module that allows benchmarking TS-Refinement against other early stopping metrics.
- `figures.ipynb`: Generate figures for the paper.
- `resnet.py` and `wide_resnet.py`: Deep learning model used, from https://github.com/uoguelph-mlrg/Cutout
