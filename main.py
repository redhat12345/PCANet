#!/usr/bin/python3.5

# This code attempts to replicate some of these results:
# https://arxiv.org/pdf/1404.3606.pdf

import os
import sys
from datetime import datetime
from subprocess import call

import numpy as np
import tensorflow as tf
from sklearn.svm import LinearSVC
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib import debug_data

from dataset_utils import load
from pcanet import PCANet


def gre_filter(datum, tensor):
    if isinstance(tensor, debug_data.InconvertibleTensorProto):
        return False
    elif (np.issubdtype(tensor.dtype, np.float) or
          np.issubdtype(tensor.dtype, np.complex) or
          np.issubdtype(tensor.dtype, np.integer)):
        return np.max(tensor) >= 65536000
    else:
        return False


def main():
    day_str = "{:%B_%d}".format(datetime.now())
    time_str = "{:%H_%M_%S}".format(datetime.now())
    day_dir = "log_data/" + day_str + "/"
    log_path = day_dir + day_str + "_" + time_str + "/"
    writer = tf.summary.FileWriter(log_path)

    # Open text editor to write description of the run and commit it
    if '--temp' not in sys.argv:
        if '-m' in sys.argv:
            m_i = sys.argv.index('-m')
            msg = sys.argv[m_i + 1]
            cmd = ['git', 'commit', '*.py', '-m', msg]
        else:
            cmd = ['git', 'commit', '*.py']

        os.environ['TF_LOG_DIR'] = log_path
        call(cmd)

    # setup the input data pipelines
    train_image_batch, train_label_batch, test_image_batch, test_label_batch, info = load('mnist')
    # train_image_batch, train_label_batch, test_image_batch, test_label_batch, info = load('cifar10')

    sess = tf.Session()
    tf.train.start_queue_runners(sess=sess)

    if '--debug' in sys.argv:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        sess.add_tensor_filter("greater_than_or_equal_to", gre_filter)

    init = tf.global_variables_initializer()

    # Hyper-params
    k1 = 7
    k2 = 7
    l1 = 8
    l2 = 8
    block_w = 4
    block_h = 4
    block_overlap = 0
    num_hist_bins = 2 ** l2
    stride_w = max(int((1 - block_overlap) * block_w), 1)
    stride_h = max(int((1 - block_overlap) * block_h), 1)
    w_steps = range(block_w, info.IMAGE_W + 1, stride_w)
    h_steps = range(block_h, info.IMAGE_H + 1, stride_h)
    print(list(w_steps))
    num_blocks = len(h_steps) * len(w_steps)

    hyperparams = {
        'l1': l1,
        'l2': l2,
        'k1': k1,
        'k2': k2,
        'num_hist_bins': num_hist_bins,
        'block_w': block_w,
        'block_h': block_h,
        'stride_w': stride_w,
        'stride_h': stride_h,
        'num_blocks': num_blocks,
    }

    # check that the blocks in the final step to be even & cover all pixels
    if w_steps[-1] != info.IMAGE_W:
        print("invalid block_overlap or block width for given image width:")
        print("W: %i, Block W: %i, Overlap: %0.2f" % (info.IMAGE_W, block_w, block_overlap))
        exit(0)

    if h_steps[-1] != info.IMAGE_H:
        print("invalid block_overlap or block height for given image height")
        print("H: %i, Block H: %i, Overlap: %0.2f" % (info.IMAGE_H, block_h, block_overlap))
        exit(0)

    # define the model
    m = PCANet(train_image_batch, hyperparams, info)

    # define placeholders for putting scores on Tensorboard
    train_score_tensor = tf.placeholder(tf.float32, shape=[], name='train_score')
    test_score_tensor = tf.placeholder(tf.float32, shape=[], name='test_score')
    tf.summary.scalar("train_score", train_score_tensor, collections=['train'])
    tf.summary.scalar("test_score", test_score_tensor, collections=['test'])

    # run it
    sess.run(init)
    writer.add_graph(sess.graph)
    merged_summary_op = tf.summary.merge_all('summaries')
    train_summary_op = tf.summary.merge_all('train')
    test_summary_op = tf.summary.merge_all('test')

    # extract PCA features from training set
    train_pcanet_features, train_labels, summary = sess.run([m.output_features, train_label_batch, merged_summary_op])
    writer.add_summary(summary, 0)

    # train linear SVM
    svm = LinearSVC(C=1.0, fit_intercept=False)
    svm.fit(train_pcanet_features, train_labels)
    train_score = svm.score(train_pcanet_features, train_labels)

    print("training score:", train_score)
    train_summary = sess.run(train_summary_op, feed_dict={train_score_tensor: train_score})
    writer.add_summary(train_summary, 0)

    # switch to test set, compute PCA filters, and score with learned SVM parameters
    scores = []
    m = PCANet(test_image_batch, hyperparams, info)
    for i in range(4):
        test_pcanet_features, test_labels, merged_summary = sess.run([m.output_features, test_label_batch, merged_summary_op])
        writer.add_summary(merged_summary, i + 1)

        score = svm.score(test_pcanet_features, test_labels)
        scores.append(score)

        print("batch test score:", score)
        test_summary = sess.run(test_summary_op, feed_dict={test_score_tensor: score})
        writer.add_summary(test_summary, i + 1)

    print("Final score on test set: ", sum(scores) / len(scores))
    writer.close()


if __name__ == '__main__':
    main()
