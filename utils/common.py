import os, cv2, ipdb
import numpy as np
import configure as cfg


def writer(msg, params, f):
    print msg % params
    msg += '\n'
    f.write(msg % params)
    return


# batch manager--------------------------------------------------------------------------
def next_batch(indices, start_idx, batch_size):
    N = len(indices)
    if start_idx+batch_size > N:
        stop_idx = N
    else:
        stop_idx = start_idx+batch_size
    return stop_idx


def early_stopping(old_val, new_val, patience_count, tolerance=1e-2, patience_limit=3):
    to_stop = False
    improvement = new_val - old_val
    if improvement < tolerance:
        if patience_count < patience_limit:
            patience_count += 1
        else:
            to_stop = true
    else:
        patience_count = 0
    return to_stop, patience_count


# data loader----------------------------------------------------------------------------
def load_frames(data_list, data_dir):
    N = len(data_list)
    data = np.zeros((N,3,cfg.TIME_S,cfg.IMG_RAW_H,cfg.IMG_RAW_W), dtype=np.uint8)
    labels = np.zeros((N,cfg.N_CATEGORIES), dtype=np.uint8)
    lim = 10
    for i in range(N):
        # parse line
        line = data_list[i]
        pth, frame_start, cat_id = line.split(' ')

        # load sequence of frames
        frame_id = int(frame_start)
        clip = np.zeros((cfg.TIME_S,3,cfg.IMG_RAW_H,cfg.IMG_RAW_W))
        for j in range(cfg.TIME_S):
            filename = '{0:06d}.jpg'.format(frame_id)
            img = cv2.imread(os.path.join(data_dir,pth,filename))
            img = img.transpose([2,0,1])
            clip[j] = img[np.newaxis, ...]
            frame_id += 1
        clip = clip.transpose([1,0,2,3])

        # append to output
        data[i] = clip[np.newaxis, ...]
        labels[i,int(cat_id)] = 1

        if i*100.0 / N > lim:
            print '    Loaded %d / %d' % (i, N)
            lim += 10
        
    return data, labels


# data helpers-------------------------------------------------------------------------
def random_crop(images):
    img_s = cfg.IMG_S
    u = np.random.randint(cfg.IMG_RAW_H - img_s + 1)
    v = np.random.randint(cfg.IMG_RAW_W - img_s + 1)
    images_crop = images[:,:,:,u:u+img_s,v:v+img_s]
    return images_crop


def central_crop(images):
    img_s = cfg.IMG_S
    u = np.random.randint(cfg.IMG_RAW_H - img_s) / 2
    v = np.random.randint(cfg.IMG_RAW_W - img_s) / 2
    images_crop = images[:,:,:,u:u+img_s,v:v+img_s]
    return images_crop

