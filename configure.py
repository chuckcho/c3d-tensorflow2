import os

# Directories
DIR_HOME    = os.path.expanduser('~')
DIR_CKPT    = 'checkpoints'
DIR_LST     = 'lists'
DIR_MODEL   = 'models'
DIR_SUMMARY = 'summary'
DIR_LOG     = 'logs'

DIR_DATA_RAW    = os.path.join(DIR_HOME, 'data', 'UCF101')
DIR_DATA        = os.path.join(DIR_HOME, 'data', 'UCF101-processed')


# Lists
PTH_TRAIN_LST   = os.path.join(DIR_LST, 'train_01.lst')
PTH_EVAL_LST    = os.path.join(DIR_LST, 'test_01.lst')
PTH_CATEGORIES  = os.path.join(DIR_LST, 'categories')

# Models
PTH_WEIGHT_C3D  = os.path.join(DIR_MODEL, 'c3d_weights.npy')
PTH_MEAN_IMG    = os.path.join(DIR_MODEL, 'train01_16_128_171_mean.npy')


# Categories
#CATEGORIES      #TODO


# Parameters
IMG_RAW_H = 128
IMG_RAW_W = 171
IMG_S = 112
TIME_S = 16
