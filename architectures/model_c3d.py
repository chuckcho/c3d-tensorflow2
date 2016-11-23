import os, ipdb
import tensorflow as tf
import numpy as np
from utils.tfcommon import *

FLAGS = tf.app.flags.FLAGS

def inference(video, net_data, keep_prob, tag=''):
    """Build the inference for one single channel.

    Args:
        video: 5D tensor: [batch_size, 3, temporal_size, image_size, image_size]
        net_data: pretrained weights from AlexNet
        keep_prob: Tensor, 0.5 if training, 1.0 otherwise

    Returns:
        prob: softmax result
    """
    tag += '_'

    stride = 1

    # conv1a layer
    # 3x3x3 kernel with stride 1
    with tf.name_scope(tag+'conv1a') as scope:
        conv1aW = tf.Variable(net_data['conv1a'][0], name='weight')
        conv1aB = tf.Variable(net_data['conv1a'][1], name='biases')
        conv1a_in = tf.nn.conv3d(video, conv1aW, strides=[1, stride, stride, stride, 1], padding='SAME')
        conv1a = tf.nn.bias_add(conv1a_in, conv1aB)
        conv1a = tf.nn.relu(conv1a, name=scope)
        #TODO: No normalization right now


    # pool1 layer
    # The first pooling layer has no temporal element
    maxpool1 = tf.nn.max_pool3d(conv1a, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 1, 1, 1], padding='SAME')

    # conv2a layer
    # 3x3x3 kernel with stride 1
    with tf.name_scope(tag+'conv2a') as scope:
        conv2aW = tf.Variable(net_data['conv2a'][0], name='weight')
        conv2aB = tf.Variable(net_data['conv2a'][1], name='biases')
        conv2a_in = tf.nn.conv3d(video, conv2aW, strides=[1, stride, stride, stride, 1], padding='SAME')
        conv2a = tf.nn.bias_add(conv2a_in, conv2aB)
        conv2a = tf.nn.relu(conv2a, name=scope)
        #TODO: No normalization right now


    # pool2 layer
    maxpool2 = tf.nn.max_pool3d(conv2a, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1], padding='SAME')

    # conv3a layer
    # 3x3x3 kernel with stride 1
    with tf.name_scope(tag+'conv3a') as scope:
        conv3aW = tf.Variable(net_data['conv3a'][0], name='weight')
        conv3aB = tf.Variable(net_data['conv3a'][1], name='biases')
        conv3a_in = tf.nn.conv3d(video, conv3aW, strides=[1, stride, stride, stride, 1], padding='SAME')
        conv3a = tf.nn.bias_add(conv3a_in, conv3aB)
        conv3a = tf.nn.relu(conv3a, name=scope)

    # conv3b layer
    # 3x3x3 kernel with stride 1
    with tf.name_scope(tag+'conv3b') as scope:
        conv3bW = tf.Variable(net_data['conv3b'][0], name='weight')
        conv3bB = tf.Variable(net_data['conv3b'][1], name='biases')
        conv3b_in = tf.nn.conv3d(video, conv3bW, strides=[1, stride, stride, stride, 1], padding='SAME')
        conv3b = tf.nn.bias_add(conv3b_in, conv3bB)
        conv3b = tf.nn.relu(conv3b, name=scope)

    # pool3 layer
    maxpool3 = tf.nn.max_pool3d(conv3b, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1], padding='SAME')

    # conv4a layer
    # 3x3x3 kernel with stride 1
    with tf.name_scope(tag+'conv4a') as scope:
        conv4aW = tf.Variable(net_data['conv4a'][0], name='weight')
        conv4aB = tf.Variable(net_data['conv4a'][1], name='biases')
        conv4a_in = tf.nn.conv3d(video, conv4aW, strides=[1, stride, stride, stride, 1], padding='SAME')
        conv4a = tf.nn.bias_add(conv4a_in, conv4aB)
        conv4a = tf.nn.relu(conv4a, name=scope)

    # conv4b layer
    # 3x3x3 kernel with stride 1
    with tf.name_scope(tag+'conv4b') as scope:
        conv4bW = tf.Variable(net_data['conv4b'][0], name='weight')
        conv4bB = tf.Variable(net_data['conv4b'][1], name='biases')
        conv4b_in = tf.nn.conv3d(video, conv4bW, strides=[1, stride, stride, stride, 1], padding='SAME')
        conv4b = tf.nn.bias_add(conv4b_in, conv4bB)
        conv4b = tf.nn.relu(conv4b, name=scope)

    # pool4 layer
    maxpool4 = tf.nn.max_pool3d(conv4b, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1], padding='SAME')

    # conv5a layer
    # 3x3x3 kernel with stride 1
    with tf.name_scope(tag+'conv5a') as scope:
        conv5aW = tf.Variable(net_data['conv5a'][0], name='weight')
        conv5aB = tf.Variable(net_data['conv5a'][1], name='biases')
        conv5a_in = tf.nn.conv3d(video, conv5aW, strides=[1, stride, stride, stride, 1], padding='SAME')
        conv5a = tf.nn.bias_add(conv5a_in, conv5aB)
        conv5a = tf.nn.relu(conv5a, name=scope)

    # conv5b layer
    # 3x3x3 kernel with stride 1
    with tf.name_scope(tag+'conv5b') as scope:
        conv5bW = tf.Variable(net_data['conv5b'][0], name='weight')
        conv5bB = tf.Variable(net_data['conv5b'][1], name='biases')
        conv5b_in = tf.nn.conv3d(video, conv5bW, strides=[1, stride, stride, stride, 1], padding='SAME')
        conv5b = tf.nn.bias_add(conv5b_in, conv5bB)
        conv5b = tf.nn.relu(conv5b, name=scope)

    # pool5 layer
    maxpool5 = tf.nn.max_pool3d(conv5b, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1], padding='SAME')

    # fc6 layer
    ## fc(4096, name='fc6')
    with tf.name_scope(tag+'fc6') as scope:
        fc6W = tf.Variable(net_data['fc6'][0], name='weight')
        fc6B = tf.Variable(net_data['fc6'][1], name='biases')
        fc6_in = tf.reshape(maxpool5, [FLAGS.batch_size, int(np.prod(maxpool5.get_shape()[1:]))])
        fc6 = tf.nn.relu_layer(fc6_in, fc6W, fc6B, name=scope)
        fc6_drop = tf.nn.dropout(fc6, keep_prob=keep_prob, name='drop')

    # fc7 layer
    ## fc(4096, name='fc7')
    with tf.name_scope(tag+'fc7') as scope:
        fc7W = tf.Variable(net_data['fc7'][0], name='weight')
        fc7B = tf.Variable(net_data['fc7'][1], name='biases')
        fc7 = tf.nn.relu_layer(fc6_drop, fc7W, fc7B, name=scope)
        fc7_drop = tf.nn.dropout(fc7, keep_prob=keep_prob, name='drop')

    # softmax layer
    prob = tf.nn.softmax(fc7_drop, name='prob')
    return prob


def loss(prob, labels, tag):
    """Return the loss as categorical cross-entropy

    Args:
        prob: results from inference
        labels: 2D Tensor [batch_size, n_classes], 1 if object in that class, 0 otherwise

    Returns:
        loss: categorical crossentropy loss
    """
    tag += '_'
    #L = -tf.reduce_sum(labels * tf.log(logits), reduction_indices=1)
    #loss = tf.reduce_sum(L, reduction_indices=0, name='loss')

    logits = tf.log(tf.clip_by_value(prob, 1e-10, 1.0), name='logits')
    L = -tf.reduce_sum(labels * logits, reduction_indices=1)
    loss = tf.reduce_sum(L, reduction_indices=0, name='loss')

    # regularize fully connected layers
    with tf.variable_scope(tag+'fc7'):
        fc7W = tf.get_variable('weight', [4096,4096], dtype=tf.float32)
        fc7b = tf.get_variable('biases', [4096], dtype=tf.float32)
    with tf.variable_scope(tag+'fc8'):
        fc8W = tf.get_variable('weight', [4096,FLAGS.n_classes], dtype=tf.float32)
        fc8b = tf.get_variable('biases', [FLAGS.n_classes], dtype=tf.float32)

    regularizers = tf.nn.l2_loss(fc7W) + tf.nn.l2_loss(fc7b) + tf.nn.l2_loss(fc8W) + tf.nn.l2_loss(fc8b)
    loss += 5e-4 * regularizers
    return loss
