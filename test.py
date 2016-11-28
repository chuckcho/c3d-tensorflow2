import configure as cfg
from utils import common
import ipdb

with open(cfg.PTH_EVAL_LST, 'r') as f: train_lst = f.read().splitlines()
train_lst = train_lst[:10] # load only 10 clips to test
data, labels = common.load_frames(train_lst, cfg.DIR_DATA)

data_crop1 = common.random_crop(data)
data_crop2 = common.random_crop(data)
ipdb.set_trace()
