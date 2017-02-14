import os

# Directories
DIR_HOME    = '/media/6TB/Videos'
DIR_CKPT    = 'checkpoints'
DIR_LST     = 'lists'
DIR_MODEL   = 'models'
DIR_SUMMARY = 'summary'
DIR_LOG     = 'logs'

# A path that contains UCF-101 videos (where you unrar'ed the original UCF-101 data)
DIR_DATA_RAW   = os.path.join(DIR_HOME, 'UCF-101')
DIR_DATA       = os.path.join(DIR_HOME, 'UCF-101')

# Lists
PTH_TRAIN_LST  = '/media/6TB/Videos/UCF-101/train_test_split/c3d_ucf101_train_split1.txt'
PTH_EVAL_LST   = '/media/6TB/Videos/UCF-101/train_test_split/c3d_ucf101_test_split1.txt'
PTH_CATEGORIES = os.path.join(DIR_LST, 'categories.lst')
#IMAGE_FORMAT   = '{:06d}.jpg'
IMAGE_FORMAT   = 'image_{:04d}.jpg'

# Models
PTH_WEIGHT_C3D = os.path.join(DIR_MODEL, 'c3d_weights.npy')
PTH_MEAN_IMG   = os.path.join(DIR_MODEL, 'train01_16_128_171_mean.npy')

# Categories
CATEGORIES     = open(PTH_CATEGORIES, 'r').read().splitlines()
N_CATEGORIES   = len(CATEGORIES)

# Parameters
IMG_RAW_H = 128
IMG_RAW_W = 171
IMG_S = 112
TIME_S = 16
