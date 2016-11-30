import os, cv2
import numpy as np
import configure as cfg
import random

def writer(msg, params, f):
    formatted_msg = msg % params
    print formatted_msg
    f.write(formatted_msg + '\n')
    return

def next_batch(num_samples, start_idx, batch_size):
    stop_idx = min(start_idx + batch_size, num_samples)
    return stop_idx

def early_stopping(
    old_val,
    new_val,
    patience_count,
    tolerance=1e-2,
    patience_limit=3 # Quite impatient
    ):
    to_stop = False
    improvement = new_val - old_val
    if improvement < tolerance:
        if patience_count < patience_limit:
            patience_count += 1
        else:
            to_stop = True
    else:
        patience_count = 0
    return to_stop, patience_count

def load_clips_labels(data_list, data_dir):
    N = len(data_list)
    clips = []
    labels = []
    for i in xrange(N):
        # Parse a line
        line = data_list[i]
        filename, frame_start, cat_id = line.split(' ')
        # Load sequence of frames
        frame_id = int(frame_start)
        clips.append((os.path.join(data_dir, filename), frame_id))
        labels.append(int(cat_id))
    return clips, np.array(labels, dtype=np.uint8)

def load_frames(clips_info):
    img_size = (cfg.IMG_RAW_H, cfg.IMG_RAW_W)
    N = len(clips_info)
    data = np.zeros(
        (N, cfg.TIME_S) + img_size + (3,),
        dtype=np.uint8)
    for clip_count, clip_info in enumerate(clips_info):
        video_path, start_frame = clip_info
        clip = np.zeros((cfg.TIME_S, 3) + img_size)
        for frame_count in range(cfg.TIME_S):
            filename = cfg.IMAGE_FORMAT.format(start_frame)
            img = cv2.imread(os.path.join(video_path, filename))
            # in case image was not resized at extraction time
            if img.shape[1:] != img_size:
                img = cv2.resize(
                    img,
                    img_size[::-1],
                    interpolation=cv2.INTER_CUBIC)
            img = img.transpose([2, 0, 1])
            clip[frame_count] = img[np.newaxis, ...]
            start_frame += 1
        clip = clip.transpose([0, 2, 3, 1])
        data[clip_count] = clip[np.newaxis, ...]
    return data

def crop_clips(clips, random_crop=True):
    img_s = cfg.IMG_S
    if random_crop:
        y = np.random.randint(cfg.IMG_RAW_H - img_s + 1)
        x = np.random.randint(cfg.IMG_RAW_W - img_s + 1)
    else:
        y = (cfg.IMG_RAW_H - img_s) / 2
        x = (cfg.IMG_RAW_W - img_s) / 2
    cropped_clips = clips[:, :, y:y+img_s, x:x+img_s, :]
    return cropped_clips
