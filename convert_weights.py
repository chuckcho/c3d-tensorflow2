#!/usr/bin/env python

import caffe
import numpy as np
import os

DIR_MODEL = 'models'

model = os.path.join(DIR_MODEL, 'conv3d_deploy.prototxt')
weights = os.path.join(DIR_MODEL, 'conv3d_deepnetA_sport1m_iter_1900000')
layers = ['conv1a', 'conv2a', 'conv3a', 'conv3b', 'conv4a', 'conv4b','conv5a', 'conv5b', 'fc6', 'fc7', 'fc8']
output = os.path.join(DIR_MODEL, 'c3d_weights.npy')

'''
conv1a: w_shape, b_shape=(64, 3, 3, 3, 3), (1, 1, 1, 1, 64)
conv2a: w_shape, b_shape=(128, 64, 3, 3, 3), (1, 1, 1, 1, 128)
conv3a: w_shape, b_shape=(256, 128, 3, 3, 3), (1, 1, 1, 1, 256)
conv3b: w_shape, b_shape=(256, 256, 3, 3, 3), (1, 1, 1, 1, 256)
conv4a: w_shape, b_shape=(512, 256, 3, 3, 3), (1, 1, 1, 1, 512)
conv4b: w_shape, b_shape=(512, 512, 3, 3, 3), (1, 1, 1, 1, 512)
conv5a: w_shape, b_shape=(512, 512, 3, 3, 3), (1, 1, 1, 1, 512)
conv5b: w_shape, b_shape=(512, 512, 3, 3, 3), (1, 1, 1, 1, 512)
fc6: w_shape, b_shape=(1, 1, 1, 4096, 8192), (1, 1, 1, 1, 4096)
fc7: w_shape, b_shape=(1, 1, 1, 4096, 4096), (1, 1, 1, 1, 4096)
fc8: w_shape, b_shape=(1, 1, 1, 101, 4096), (1, 1, 1, 1, 101)
'''

def main():
    # Per https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#conv3d
    # Filter has shape: [filter_depth, filter_height, filter_width, in_channels, out_channels]
    net = caffe.Net(model, weights)
    netdata = dict()
    for layer in layers:
        print "{}: w_shape, b_shape={}, {}".format(layer, net.params[layer][0].data.shape, net.params[layer][1].data.shape)
        # per https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/api_docs/python/functions_and_classes/shard5/tf.nn.conv3d.md
        # filter: A Tensor. Must have the same type as input. Shape [filter_depth, filter_height, filter_width, in_channels, out_channels]. in_channels must match between input and filter.
        w = np.transpose(net.params[layer][0].data, (2, 4, 3, 1, 0))
        b = np.squeeze(net.params[layer][1].data)
        netdata.update({layer: (w, b)})
    np.save(output, netdata)

if __name__ == '__main__':
    main()
