import numpy as np
import cv2
import configure as cfg
import os, glob, ipdb


def create_dirs():
    if not os.path.exists(cfg.DIR_DATA): os.makedirs(cfg.DIR_DATA)
    for category in cfg.CATEGORIES:
        pth = os.path.join(cfg.DIR_DATA, category)
        if not os.path.exists(pth): os.makedirs(pth)
    return


def preprocess():
    lst = os.listdir(cfg.DIR_DATA_RAW)
    newsize = (cfg.IMG_RAW_W, cfg.IMG_RAW_H)

    for filename in lst:
        print filename

        # parse file name
        instance = filename.replace('.avi','')
        category = instance[instance.index('_')+1:]
        category = category[:category.index('_')]

        # create new directory if needed
        pth = os.path.join(cfg.DIR_DATA, category, instance)
        if not os.path.exists(pth): os.makedirs(pth)

        # read video and extract frame
        cap = cv2.VideoCapture(os.path.join(cfg.DIR_DATA_RAW, filename))
        frameid = 0
        while cap.isOpened():
            # read new frame
            ret, frame = cap.read()
            if ret == False: break # end of video

            # resize and write frame
            frame = cv2.resize(frame, newsize, interpolation=cv2.INTER_CUBIC)
            frame_file = os.path.join(pth, '{0:06d}.jpg'.format(frameid+1))
            cv2.imwrite(frame_file, frame)
            frameid += 1
        cap.release()

    return


if __name__ == '__main__':
    create_dirs()
    preprocess()
