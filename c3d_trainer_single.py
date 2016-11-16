import tensorflow as tf
import numpy as np
import model_single_channel as model
import os, sys, time, ipdb
from utils import common
import configure as cfg


# model parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_iter', 100, """Maximum number of training iteration.""")
tf.app.flags.DEFINE_integer('batch_size', 30, """Numer of videos to process in a batch.""")
#tf.app.flags.DEFINE_integer('img_s', cfg.IMG_S, """"Size of a square image.""")
tf.app.flags.DEFINE_integer('n_classes', 101, """Number of classes.""")
tf.app.flags.DEFINE_float('learning_rate', 0.003, """"Learning rate for training models.""")
tf.app.flags.DEFINE_integer('summary_frequency', 10, """How often to write summary.""")
tf.app.flags.DEFINE_integer('checkpoint_frequency', 10, """How often to evaluate and write checkpoint.""")

#=========================================================================================
def placeholder_inputs(batch_size):
    """Create placeholders for tensorflow with some specific batch_size

    Args:
        batch_size: size of each batch

    Returns:
        videos_ph: 5D tensor of shape [batch_size, temporal_dimension, width, height, 3]
        labels_ph: 2D tensor of shape [batch_size, num_classes]
        keep_prob_ph: 1D tensor for the keep_probability (for dropout during training)
    """
    videos_ph = tf.placeholder(tf.float32, shape=(batch_size, 16, 128, 171, 3), name='videos_placeholder')
    labels_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.n_classes), name='labels_placeholder')
    keep_prob_ph = tf.placeholder(tf.float32, shape=(), name='keep_prob_placeholder')

    return videos_ph, labels_ph, keep_prob_ph
