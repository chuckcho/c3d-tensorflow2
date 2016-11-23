import caffe
import numpy as np
import os, ipdb


DIR_MODEL = 'models'

model = os.path.join(DIR_MODEL, 'conv3d_deploy.prototxt')
weights = os.path.join(DIR_MODEL, 'conv3d_deepnetA_sport1m_iter_1900000')
layers = ['conv1a', 'conv2a', 'conv3a', 'conv3b', 'conv4a', 'conv4b','conv5a', 'conv5b', 'fc6', 'fc7', 'fc8']
output = os.path.join(DIR_MODEL, 'c3d_weights.npy')


if __name__ == '__main__':
    net = caffe.Net(model, weights)
    netdata = {layer: (net.params[layer][0].data, net.params[layer][1].data) for layer in layers}
    np.save(output, netdata)
