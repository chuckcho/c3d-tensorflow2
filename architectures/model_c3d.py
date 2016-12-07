import os
import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def max_pool(in_data, k=2, name=None):
    # 3d maxpool with k temporal kernel/stride
    return tf.nn.max_pool3d(
        in_data,
        ksize=[1, k, 2, 2, 1],
        strides=[1, k, 2, 2, 1],
        padding='SAME',
        name=name)

def inference(video, net_data, keep_prob, tag=''):
    """Build the inference

    Args:
        video: 5D tensor:
            [batch_size, 3, temporal_size, image_size, image_size]
        net_data: trained weights
        keep_prob: Tensor, 0.5 if training, 1.0 otherwise for dropout layers

    Returns:
        logits: result (before softmax)
    """
    tag += '_'

    stride = 1
    # Common 5D stride
    strides = [1, stride, stride, stride, 1]

    # Conv1a layer
    with tf.name_scope(tag+'conv1a') as scope:
        conv1aW = tf.Variable(net_data['conv1a'][0], name='weight')
        conv1aB = tf.Variable(net_data['conv1a'][1], name='biases')
        conv1a_in = tf.nn.conv3d(video, conv1aW, strides=strides,
                                 padding='SAME')
        conv1a = tf.nn.bias_add(conv1a_in, conv1aB)
        conv1a = tf.nn.relu(conv1a, name=scope)

    # Pool1 layer
    maxpool1 = max_pool(conv1a, k=1, name='pool1')

    # Conv2a layer
    with tf.name_scope(tag+'conv2a') as scope:
        conv2aW = tf.Variable(net_data['conv2a'][0], name='weight')
        conv2aB = tf.Variable(net_data['conv2a'][1], name='biases')
        conv2a_in = tf.nn.conv3d(maxpool1, conv2aW, strides=strides,
                                 padding='SAME')
        conv2a = tf.nn.bias_add(conv2a_in, conv2aB)
        conv2a = tf.nn.relu(conv2a, name=scope)

    # Pool2 layer
    maxpool2 = max_pool(conv2a, name='pool2')

    # Conv3a layer
    with tf.name_scope(tag+'conv3a') as scope:
        conv3aW = tf.Variable(net_data['conv3a'][0], name='weight')
        conv3aB = tf.Variable(net_data['conv3a'][1], name='biases')
        conv3a_in = tf.nn.conv3d(maxpool2, conv3aW, strides=strides,
                                 padding='SAME')
        conv3a = tf.nn.bias_add(conv3a_in, conv3aB)
        conv3a = tf.nn.relu(conv3a, name=scope)

    # Conv3b layer
    with tf.name_scope(tag+'conv3b') as scope:
        conv3bW = tf.Variable(net_data['conv3b'][0], name='weight')
        conv3bB = tf.Variable(net_data['conv3b'][1], name='biases')
        conv3b_in = tf.nn.conv3d(conv3a, conv3bW, strides=strides,
                                 padding='SAME')
        conv3b = tf.nn.bias_add(conv3b_in, conv3bB)
        conv3b = tf.nn.relu(conv3b, name=scope)

    # Pool3 layer
    maxpool3 = max_pool(conv3b, name='pool3')

    # Conv4a layer
    with tf.name_scope(tag+'conv4a') as scope:
        conv4aW = tf.Variable(net_data['conv4a'][0], name='weight')
        conv4aB = tf.Variable(net_data['conv4a'][1], name='biases')
        conv4a_in = tf.nn.conv3d(maxpool3, conv4aW, strides=strides,
                                 padding='SAME')
        conv4a = tf.nn.bias_add(conv4a_in, conv4aB)
        conv4a = tf.nn.relu(conv4a, name=scope)

    # Conv4b layer
    with tf.name_scope(tag+'conv4b') as scope:
        conv4bW = tf.Variable(net_data['conv4b'][0], name='weight')
        conv4bB = tf.Variable(net_data['conv4b'][1], name='biases')
        conv4b_in = tf.nn.conv3d(conv4a, conv4bW, strides=strides,
                                 padding='SAME')
        conv4b = tf.nn.bias_add(conv4b_in, conv4bB)
        conv4b = tf.nn.relu(conv4b, name=scope)

    # Pool4 layer
    maxpool4 = max_pool(conv4b, name='pool4')

    # Conv5a layer
    with tf.name_scope(tag+'conv5a') as scope:
        conv5aW = tf.Variable(net_data['conv5a'][0], name='weight')
        conv5aB = tf.Variable(net_data['conv5a'][1], name='biases')
        conv5a_in = tf.nn.conv3d(maxpool4, conv5aW, strides=strides,
                                 padding='SAME')
        conv5a = tf.nn.bias_add(conv5a_in, conv5aB)
        conv5a = tf.nn.relu(conv5a, name=scope)

    # Conv5b layer
    with tf.name_scope(tag+'conv5b') as scope:
        conv5bW = tf.Variable(net_data['conv5b'][0], name='weight')
        conv5bB = tf.Variable(net_data['conv5b'][1], name='biases')
        conv5b_in = tf.nn.conv3d(conv5a, conv5bW, strides=strides,
                                 padding='SAME')
        conv5b = tf.nn.bias_add(conv5b_in, conv5bB)
        conv5b = tf.nn.relu(conv5b, name=scope)

    # Pool5 layer
    maxpool5 = max_pool(conv5b, name='pool5')

    # Fc6 layer
    with tf.name_scope(tag+'fc6') as scope:
        fc6W = tf.Variable(net_data['fc6'][0], name='weight')
        fc6B = tf.Variable(net_data['fc6'][1], name='biases')
        fc6_in = tf.reshape(
            maxpool5,
            [FLAGS.batch_size, int(np.prod(maxpool5.get_shape()[1:]))])
        fc6 = tf.nn.relu_layer(fc6_in, fc6W, fc6B, name=scope)
        fc6_drop = tf.nn.dropout(fc6, keep_prob=keep_prob, name='drop')

    # Fc7 layer
    with tf.name_scope(tag+'fc7') as scope:
        fc7W = tf.Variable(net_data['fc7'][0], name='weight')
        fc7B = tf.Variable(net_data['fc7'][1], name='biases')
        fc7 = tf.nn.relu_layer(fc6_drop, fc7W, fc7B, name=scope)
        fc7_drop = tf.nn.dropout(fc7, keep_prob=keep_prob, name='drop')

    # Fc8 layer
    with tf.name_scope(tag+'fc8') as scope:
        fc8W = tf.Variable(net_data['fc8'][0], name='weight')
        fc8B = tf.Variable(net_data['fc8'][1], name='biases')
        #fc8 = tf.nn.relu_layer(fc7_drop, fc8W, fc8B, name=scope)
        #fc8_drop = tf.nn.dropout(fc8, keep_prob=keep_prob, name='drop')

    # Return logits before softmax
    #logits = fc8_drop
    logits = tf.matmul(fc7_drop, fc8W) + fc8B

    # Skipping softmax layer
    # prob = tf.nn.softmax(fc8_drop, name='prob')
    return logits

def loss(logits, labels, tag):
    """Return the loss as categorical cross-entropy

    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size]

    Returns:
        loss: Loss tensor of type float.
    """
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits,
            labels,
            name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    return loss

def training(loss, learning_rate):
  """Sets up the training Ops.
  Creates a summarizer to track the loss over time in TensorBoard.
  Creates an optimizer and applies the gradients to all trainable variables.
  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.
  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A int32 tensor with each entry being 0 (predicted incorrectly)
    or 1 (predicted correctly)
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  return tf.cast(correct, tf.int32)
