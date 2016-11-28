import os, ipdb
import numpy as np
import configure as cfg


def gen_dict():
    lst = os.listdir(cfg.DIR_DATA_RAW)
    categories = [item[item.index('_')+1:] for item in lst]
    categories = [item[:item.index('_')] for item in categories]
    categories = np.unique(categories)

    f = open(cfg.PTH_CATEGORIES, 'w')
    for item in categories:
        f.write(item+'\n')
    f.close()
    return


if __name__ == '__main__':
    gen_dict()
