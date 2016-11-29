It's still work in progress.

1. Download UCF-101 dataset from [UCF-101 website](http://crcv.ucf.edu/data/UCF101.php).
2. Unzip the dataset: e.g. `unrar x UCF101.rar`
3. Extract frames from UCF-101 videos by revising and running a helper script, `extract_frames.sh`.
4. Download pre-trained weights and mean cube files: `bash get_weights_and_mean.sh`
5. Add facebook/C3D python path: `export PYTHONPATH=$PYTHONPATH:/path/to/facebook-c3d/python`
6. Convert weights: `python convert_weights.py`
7. Convert a mean cube: `python convert_mean.py`
