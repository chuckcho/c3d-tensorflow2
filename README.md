C3D-TensorFlow2
===============

This was forked from https://github.com/toddnguyen/c3d-tensorflow, but debugged/modified significantly to work with existing UCF-101 data set and in dextro environment.
It's a WIP for end-to-end training for [8-conv layer C3D network](https://github.com/facebook/C3D/blob/master/examples/c3d_finetuning/c3d_ucf101_finetuning_train.prototxt).

## Training
Steps to start raining:

1. Download UCF-101 dataset from [UCF-101 website](http://crcv.ucf.edu/data/UCF101.php).
2. Unzip the dataset: e.g. `unrar x UCF101.rar`
3. Extract frames from UCF-101 videos by revising and running a helper script, `bash tools/extract_frames.sh`.
4. Download pre-trained weights and mean cube files: `bash models/get_weights_and_mean.sh`
5. Add facebook/C3D python path (requirement for the subsequent steps): `export PYTHONPATH=$PYTHONPATH:/path/to/facebook-c3d/python`
6. Convert weights: `python convert_weights.py`
7. Convert a mean cube: `python convert_mean.py`
