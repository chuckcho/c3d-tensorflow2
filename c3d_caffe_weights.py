import caffe
import numpy

net = caffe.Net('c3d_ucf101_finetuning_train.prototxt', 'conv3d_deepnetA_sport1m_iter_1900000')

params = ['conv1a', 'conv2a', 'conv3a', 'conv3b', 'conv4a', 'conv4b','conv5a', 'conv5b', 'fc6', 'fc7', 'fc8']

# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data)

for pr in params:
    print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)
