import caffe
import numpy as np
import os


DIR_MODEL = 'models'

model = os.path.join(DIR_MODEL, 'conv3d_deploy.prototxt')
weights = os.path.join(DIR_MODEL, 'conv3d_deepnetA_sport1m_iter_1900000')
layers = ['conv1a', 'conv2a', 'conv3a', 'conv3b', 'conv4a', 'conv4b','conv5a', 'conv5b', 'fc6', 'fc7', 'fc8']
output = os.path.join(DIR_MODEL, 'c3d_weights.npy')


if __name__ == '__main__':
    # Per https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#conv3d
    # Filter has shape: [filter_depth, filter_height, filter_width, in_channels, out_channels]
    net = caffe.Net(model, weights)
    netdata = {
            layer: (
                    np.transpose(net.params[layer][0].data, (2, 4, 3, 1, 0)),
                    np.squeeze(net.params[layer][1].data)
                    ) for layer in layers
            }
    np.save(output, netdata)
