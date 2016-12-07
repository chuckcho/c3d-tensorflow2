#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import architectures.model_c3d as model
import os
import sys
import time
from utils import common
import configure as cfg

# Training parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer(
    'max_epoch', 10, "Maximum number of training epochs.")
tf.app.flags.DEFINE_integer(
    'batch_size', 16, "Number of videos to process in a batch.")
tf.app.flags.DEFINE_integer(
    'img_s', cfg.IMG_S, "Size of a square image.")
tf.app.flags.DEFINE_integer(
    'time_s', cfg.TIME_S, "Temporal length.")
tf.app.flags.DEFINE_integer(
    'num_classes', cfg.N_CATEGORIES, "Number of classes.")
tf.app.flags.DEFINE_float(
    'learning_rate', 0.003, "Learning rate for training models.")
tf.app.flags.DEFINE_integer(
    'summary_frequency', 100, "How often in iterations to write summary.")
tf.app.flags.DEFINE_integer(
    'checkpoint_frequency', 1000,
    "How often in iteraions to save checkpoint.")
tf.app.flags.DEFINE_integer(
    'validation_frequency', 10,
    "How often in iteraions to evaluate.")

def placeholder_inputs(batch_size):
    # Per https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#conv3d
    # Input: A Tensor with shape:
    # [batch, in_depth, in_height, in_width, in_channels].
    videos_ph = tf.placeholder(
            tf.float32,
            shape=(None, FLAGS.time_s, FLAGS.img_s, FLAGS.img_s, 3),
            name='videos_placeholder')
    labels_ph = tf.placeholder(
            tf.int32,
            shape=(None),
            name='labels_placeholder')
    # for dropout layers
    keep_prob_ph = tf.placeholder(
            tf.float32,
            shape=(),
            name='keep_prob_placeholder')
    return videos_ph, labels_ph, keep_prob_ph

def fill_feed_dict(
        videos_batch,
        labels_batch,
        videos_ph,
        labels_ph,
        keep_prob_ph,
        is_training=False):
    # Read frames
    video_data = common.load_frames(videos_batch)

    # Fill (zero-pad) up to the batch size
    if labels_batch.shape[0] < FLAGS.batch_size:
        M = FLAGS.batch_size - labels_batch.shape[0]
        video_data = np.pad(
                video_data,
                ((0, M), (0, 0), (0, 0), (0, 0), (0, 0)),
                'constant',
                constant_values=0)
        labels_batch = np.pad(
                labels_batch,
                (0, M),
                'constant',
                constant_values=0)
    if is_training:
        feed_dict = {
                videos_ph: common.crop_clips(video_data, random_crop=True),
                labels_ph: labels_batch,
                keep_prob_ph: 0.5 # in training time, drop out with 1/2 prob
                }
    else:
        feed_dict = {
                videos_ph: common.crop_clips(video_data, random_crop=False),
                labels_ph: labels_batch,
                keep_prob_ph: 1.0 # in testing time, don't drop out
                }
    return feed_dict

def do_eval(
        sess,
        eval_correct,
        videos_ph,
        labels_ph,
        keep_prob_ph,
        all_data,
        all_labels,
        logfile,
        eval_subsample_rate=1):
    # Initialize
    true_count, start_idx = 0, 0

    # Subsamples
    if eval_subsample_rate > 1:
        all_data = all_data[::eval_subsample_rate]
        all_labels = all_labels[::eval_subsample_rate]

    num_samples = all_labels.shape[0]
    indices = np.arange(num_samples)
    while start_idx < num_samples:
        stop_idx = common.next_batch(num_samples, start_idx, FLAGS.batch_size)
        batch_idx = indices[start_idx:stop_idx]
        fd = fill_feed_dict(
            slice_list(all_data, batch_idx),
            all_labels[batch_idx],
            videos_ph,
            labels_ph,
            keep_prob_ph,
            is_training=False)
        tmp_ = sess.run(eval_correct, feed_dict=fd)
        # If data was partially filled in a batch, count only those
        num_valid = stop_idx - start_idx
        tmp_ = tmp_[0:num_valid]
        print "[Debug] eval_correct={}".format(tmp_)
        true_count_per_batch = sess.run(tf.reduce_sum(tmp_))
        true_count += true_count_per_batch
        start_idx = stop_idx

    precision = float(true_count) / num_samples
    common.writer("[Info] #samples=%d, #correct=%d (accuracy=%0.02f)",
                  (num_samples, true_count, precision),
                  logfile)
    return precision

def slice_list(in_list, np_array_indices):
    return [x for i, x in enumerate(in_list) if i in np_array_indices]

def run_training(
        pth_train_lst,
        train_dir,
        pth_eval_lst,
        eval_dir,
        tag):

    # For training, subsample training/eval data sets (if >1)
    subsample_rate = 1000
    # For periodic evaluation, subsample training/eval data sets (if >1)
    eval_subsample_rate = 10

    # mkdir
    if not os.path.exists(cfg.DIR_LOG):
        os.makedirs(cfg.DIR_LOG)
    if not os.path.exists(cfg.DIR_CKPT):
        os.makedirs(cfg.DIR_CKPT)
    logfile = open(
        os.path.join(cfg.DIR_LOG, "training_{}.log".format(tag)), 'w', 0)

    # Load model
    print "[Info] loading pre-trained model..."
    net_data = np.load(cfg.PTH_WEIGHT_C3D).item()
    for k in sorted(net_data.keys()):
        layer = net_data[k]
        print "[Info] layer {}: weight_shape={}, bias_shape={}".format(
            k,
            layer[0].shape,
            layer[1].shape,
            )

    # Load train/eval lists
    print "[Info] loading lists..."
    with open(pth_train_lst, 'r') as f:
        train_lst = f.read().splitlines()
    with open(pth_eval_lst,  'r') as f:
        eval_lst  = f.read().splitlines()

    # Sample train/eval instances
    if subsample_rate > 1:
        train_lst = train_lst[::subsample_rate]
        eval_lst = eval_lst[::subsample_rate]

    print "[Info] loading training data..."
    train_clips, train_labels = common.load_clips_labels(train_lst, train_dir)
    num_train = len(train_clips)

    print "[Info] loading validation data..."
    eval_clips, eval_labels = common.load_clips_labels(eval_lst, eval_dir)

    # TensorFlow variables and operations
    print "[Info] preparing tensorflow..."
    videos_ph, labels_ph, keep_prob_ph = placeholder_inputs(FLAGS.batch_size)

    # Sets up model inference, training, etc
    print "[Info] setting up model..."
    logits = model.inference(videos_ph, net_data, keep_prob_ph, tag)
    loss = model.loss(logits, labels_ph, tag)
    train_op = model.training(loss, learning_rate=FLAGS.learning_rate)
    eval_correct = model.evaluation(logits, labels_ph)
    init_op = tf.initialize_all_variables()

    # TensorFlow monitor
    print "[Info] setting up monitor..."
    summary = tf.merge_all_summaries()
    saver = tf.train.Saver()

    # Initialize graph
    print "[Info] initializing session..."
    sess = tf.Session()
    summary_writer = tf.train.SummaryWriter(cfg.DIR_SUMMARY, sess.graph)
    sess.run(init_op)

    # Start the training loop
    old_precision = sys.maxsize
    patience_count = 0
    print "[Info] starting the training loop..."

    iters = 0
    for epoch in range(FLAGS.max_epoch):
        start_time = time.time()

        # Shuffle indices
        print "[Info] starting a new epoch: shuffling indices..."
        indices = np.random.permutation(num_train)

        # Train for one epoch
        total_loss, start_idx = 0.0, 0
        while start_idx < num_train:
            stop_idx = common.next_batch(num_train, start_idx, FLAGS.batch_size)
            print "[Info] iteration {}: training {}/{} examples".format(
                iters,
                stop_idx,
                num_train)
            batch_idx = indices[start_idx:stop_idx]
            fd = fill_feed_dict(
                slice_list(train_clips, batch_idx),
                train_labels[batch_idx],
                videos_ph,
                labels_ph,
                keep_prob_ph,
                is_training=True)
            _, loss_value = sess.run([train_op, loss], feed_dict=fd)
            assert not np.isnan(loss_value), "Loss value is NaN"
            total_loss += loss_value
            iters += 1
            start_idx = stop_idx

            # Write summary
            if iters % FLAGS.summary_frequency == 0:
                duration = time.time() - start_time
                common.writer("[Info] iteration %d: loss = %.2f (%.1f sec)",
                              (iters, total_loss, duration),
                              logfile)
                summary_str = sess.run(summary, feed_dict=fd)
                summary_writer.add_summary(summary_str, epoch)
                summary_writer.flush()
                start_time = time.time()

            if iters % FLAGS.validation_frequency == 0:
                common.writer("[Info] training data eval:", (), logfile)
                do_eval(sess,
                        eval_correct,
                        videos_ph,
                        labels_ph,
                        keep_prob_ph,
                        train_clips,
                        train_labels,
                        logfile,
                        eval_subsample_rate=eval_subsample_rate)
                common.writer("[Info] validation data eval:", (), logfile)
                precision = do_eval(
                    sess,
                    eval_correct,
                    videos_ph,
                    labels_ph,
                    keep_prob_ph,
                    eval_clips,
                    eval_labels,
                    logfile,
                    eval_subsample_rate=eval_subsample_rate)
                common.writer("[Info] accuracy: %.3f", precision, logfile)

            # Write checkpoint
            if iters % FLAGS.checkpoint_frequency == 0 or \
                epoch == FLAGS.max_epoch - 1:
                checkpoint_file = os.path.join(cfg.DIR_CKPT, tag)
                saver.save(sess, checkpoint_file, global_step=epoch)

        # Early stop
        '''
        to_stop, patience_count = common.early_stopping(
                old_precision,
                precision,
                patience_count,
                patience_limit=99999
                )

        old_precision = precision
        if to_stop:
            common.writer("[Info] early stopping...", (), logfile)
            break
        '''
    logfile.close()
    return

def main(argv=None):
    with tf.Graph().as_default():
        run_training(
                cfg.PTH_TRAIN_LST,
                cfg.DIR_DATA,
                cfg.PTH_EVAL_LST,
                cfg.DIR_DATA,
                tag='c3d')
    return

if __name__ == '__main__':
    tf.app.run(main)
