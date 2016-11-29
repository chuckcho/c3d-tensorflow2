import tensorflow as tf
import numpy as np
import architectures.model_c3d as model
import os, sys, time
from utils import common
import configure as cfg


# model parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('max_iter', 20000, """Maximum number of training iteration.""")
tf.app.flags.DEFINE_integer('batch_size', 10, """Numer of videos to process in a batch.""")
tf.app.flags.DEFINE_integer('img_s', cfg.IMG_S, """"Size of a square image.""")
tf.app.flags.DEFINE_integer('time_s', cfg.TIME_S, """Temporal length.""")
tf.app.flags.DEFINE_integer('n_classes', 101, """Number of classes.""")
tf.app.flags.DEFINE_float('learning_rate', 0.003, """"Learning rate for training models.""")
tf.app.flags.DEFINE_integer('summary_frequency', 10, """How often to write summary.""")
tf.app.flags.DEFINE_integer('checkpoint_frequency', 10, """How often to evaluate and write checkpoint.""")


#=========================================================================================
def placeholder_inputs(batch_size):
    # Per https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#conv3d
    # input: A Tensor with shape [batch, in_depth, in_height, in_width, in_channels].
    videos_ph = tf.placeholder(tf.float32, shape=(batch_size, FLAGS.time_s, FLAGS.img_s, FLAGS.img_s, 3), name='videos_placeholder')
    #labels_ph = tf.placeholder(tf.int32, shape=(batch_size, FLAGS.n_classes), name='labels_placeholder')
    labels_ph = tf.placeholder(tf.int32, shape=(batch_size), name='labels_placeholder')
    keep_prob_ph = tf.placeholder(tf.float32, shape=(), name='keep_prob_placeholder')

    return videos_ph, labels_ph, keep_prob_ph


def fill_feed_dict(videos_batch, labels_batch, videos_ph, labels_ph, keep_prob_ph, is_training):
    video_data = common.load_frames(videos_batch)

    if labels_batch.shape[0] < FLAGS.batch_size:
        M = FLAGS.batch_size - labels_batch.shape[0]
        video_data = np.pad(video_data, ((0,M),(0,0),(0,0),(0,0),(0,0)), 'constant', constant_values=0)
        labels_batch = np.pad(labels_batch, (0,M), 'constant', constant_values=0)

    if is_training:
        feed_dict = {\
                videos_ph: common.crop_clips(video_data, random_crop=True), \
                labels_ph: labels_batch, keep_prob_ph: 0.5}
    else:
        feed_dict = {\
                videos_ph: common.crop_clips(video_data, random_crop=False), \
                labels_ph: labels_batch, keep_prob_ph: 1.0}
    return feed_dict


def do_eval(sess, eval_correct, videos_ph, labels_ph, keep_prob_ph, all_data, all_labels, logfile):
    true_count, start_idx = 0, 0
    num_samples = all_labels.shape[0]
    indices = np.arange(num_samples)
    while start_idx != num_samples:
        stop_idx = common.next_batch(indices, start_idx, FLAGS.batch_size)
        batch_idx = indices[start_idx:stop_idx]
        fd = fill_feed_dict(
            slice_list(all_data,batch_idx), all_labels[batch_idx],
            videos_ph, labels_ph, keep_prob_ph, is_training=False)
        true_count += sess.run(eval_correct, feed_dict=fd)
        start_idx = stop_idx

    precision = true_count*1.0 / num_samples
    common.writer('    Num-samples:%d  Num-correct:%d  Precision:%0.04f', (num_samples, true_count, precision), logfile)
    return precision


def slice_list(in_list, np_array_indices):
    return [_X_ for _I_,_X_ in enumerate(in_list) if _I_ in np_array_indices]

#=========================================================================================
def run_training(pth_train_lst, train_dir, pth_eval_lst, eval_dir, tag):
    if not os.path.exists(cfg.DIR_LOG):
        os.makedirs(cfg.DIR_LOG)
    if not os.path.exists(cfg.DIR_CKPT):
        os.makedirs(cfg.DIR_CKPT)
    subsample_rate = 1000

    logfile = open(os.path.join(cfg.DIR_LOG, 'training_'+tag+'.log'), 'w', 0)

    # load data
    print 'Loading pretrained data...'
    net_data = np.load(cfg.PTH_WEIGHT_C3D).item()
    for k in sorted(net_data.keys()):
        layer = net_data[k]
        print "[Info] layer={}: weight_shape={}, bias_shape={}".format(
                k,
                layer[0].shape,
                layer[1].shape,
                )

    print 'Loading lists...'
    with open(pth_train_lst, 'r') as f: train_lst = f.read().splitlines()
    with open(pth_eval_lst,  'r') as f: eval_lst  = f.read().splitlines()
    if subsample_rate > 1:
        train_lst = train_lst[::subsample_rate]
        eval_lst = eval_lst[::subsample_rate]

    print 'Loading training data...'
    train_clips, train_labels = common.load_clips_labels(train_lst, train_dir)
    num_train = len(train_lst)

    print 'Loading validation data...'
    eval_clips, eval_labels = common.load_clips_labels(eval_lst, eval_dir)

    # tensorflow variables and operations
    print 'Preparing tensorflow...'
    videos_ph, labels_ph, keep_prob_ph = placeholder_inputs(FLAGS.batch_size)

    logits = model.inference(videos_ph, net_data, keep_prob_ph, tag)
    print "logits={}, labels_ph={}".format(logits, labels_ph)
    loss = model.loss(logits, labels_ph, tag)
    train_op = model.training(loss, learning_rate=FLAGS.learning_rate)

    eval_correct = model.evaluation(logits, labels_ph)
    init_op = tf.initialize_all_variables()

    # tensorflow monitor
    summary = tf.merge_all_summaries()
    saver = tf.train.Saver()

    # initialize graph
    sess = tf.Session()
    summary_writer = tf.train.SummaryWriter(cfg.DIR_SUMMARY, sess.graph)
    sess.run(init_op)

    # start the training loop
    old_precision = sys.maxsize
    patience_count = 0
    print 'Start the training loop...'
    for step in range(FLAGS.max_iter):
        # training phase-----------------------------------------------
        start_time = time.time()

        # shuffle indices
        indices = np.random.permutation(num_train)

        # train by batches
        total_loss, start_idx, lim = 0, 0, 10
        while start_idx != num_train:
            if start_idx*100.0 / num_train > lim:
                print 'Trained %d/%d' % (start_idx, num_train)
                lim += 10

            stop_idx = common.next_batch(indices, start_idx, FLAGS.batch_size)
            batch_idx = indices[start_idx: stop_idx]

            fd = fill_feed_dict(
                    slice_list(train_clips,batch_idx), train_labels[batch_idx],
                    videos_ph, labels_ph, keep_prob_ph,
                    is_training=True)
            _, loss_value = sess.run([train_op, loss], feed_dict=fd)
            assert not np.isnan(loss_value), 'Loss value is NaN'

            total_loss += loss_value
            start_idx = stop_idx
        duration = time.time() - start_time

        # write summary-----------------------------------------------
        if step % FLAGS.summary_frequency == 0:
            common.writer('Step %d: loss = %.3f (%.3f sec)', (step, total_loss, duration), logfile)
            summary_str = sess.run(summary, feed_dict=fd)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
        else:
            common.writer('Step %d', step, logfile)

        # write checkpoint---------------------------------------------
        if step % FLAGS.checkpoint_frequency == 0 or (step+1) == FLAGS.max_iter:
            checkpoint_file = os.path.join(cfg.DIR_CKPT, tag)
            print "[Debug] checkpoint_file={}, step={}".format(
                    checkpoint_file,
                    step,
                    )
            #saver.save(sess, checkpoint_file. global_step=step)

            common.writer('  Training data eval:', (), logfile)
            do_eval(
                    sess, eval_correct,
                    videos_ph, labels_ph, keep_prob_ph,
                    train_clips, train_labels, logfile)

            common.writer('  Validation data eval:', (), logfile)
            precision = do_eval(
                    sess, eval_correct,
                    videos_ph, labels_ph, keep_prob_ph,
                    eval_clips, eval_labels, logfile)
            common.writer('Precision: %.4f', precision, logfile)

        # early stopping-------------------------------------------------
        to_stop, patience_count = common.early_stopping(old_precision, precision, patience_count)
        old_precision = precision
        if to_stop:
            common.writer('Early stopping...', (), logfile)
            break
    logfile.close()
    return



#=========================================================================================
def main(argv=None):
    pth_train_lst = cfg.PTH_TRAIN_LST
    pth_eval_lst  = cfg.PTH_EVAL_LST
    train_dir = cfg.DIR_DATA
    eval_dir  = cfg.DIR_DATA
    with tf.Graph().as_default():
        run_training(pth_train_lst, train_dir, pth_eval_lst, eval_dir, tag='c3d')
    return


if __name__ == '__main__':
    tf.app.run(main)
