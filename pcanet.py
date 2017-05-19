#!/usr/bin/python3.5

# This code attempts to replicate some of these results:
# https://arxiv.org/pdf/1404.3606.pdf

import os
import sys
from datetime import datetime
from sklearn.svm import LinearSVC
from subprocess import call

import numpy as np
import tensorflow as tf

from dataset_utils import load


class PCANet:
    def __init__(self, init_image_batch, hyperparams, info):
        self._image_batch = init_image_batch
        tf.summary.image('input', self._image_batch, max_outputs=5)

        k1 = hyperparams['k1']
        k2 = hyperparams['k2']
        l1 = hyperparams['l1']
        l2 = hyperparams['l2']
        stride_w = hyperparams['stride_w']
        stride_h = hyperparams['stride_h']
        block_w = hyperparams['block_w']
        block_h = hyperparams['block_h']
        num_hist_bins = hyperparams['num_hist_bins']
        num_blocks = hyperparams['num_blocks']

        with tf.name_scope("extract_patches1"):
            self.patches1 = tf.extract_image_patches(self._image_batch, [1, k1, k2, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME',
                                                     name='patches')
            self.patches1 = tf.reshape(self.patches1, [-1, k1 * k2, info.N_CHANNELS], name='patches_shaped')
            # TODO: figure out how to unvectorize for multi-channel images
            # self.patches1 = tf.reshape(self.patches1, [-1, info.N_CHANNELS,  k1 * k2], name='patches_shaped')
            # self.patches1 = tf.transpose(self.patches1, [0, 2, 1])
            self.zero_mean_patches1 = self.patches1 - tf.reduce_mean(self.patches1, axis=1, keep_dims=True, name='patch_means')
            x1 = tf.transpose(self.zero_mean_patches1, [2, 1, 0])
            x1_trans = tf.transpose(self.zero_mean_patches1, [2, 0, 1])
            self.patches_covariance1 = tf.matmul(x1, x1_trans, name='patch_covariance')

        with tf.name_scope("eignvalue_decomposition1"):
            self.x_eig_vals1, self.x_eig1 = tf.self_adjoint_eig(self.patches_covariance1, name='x_eig')
            self.top_x_eig1 = tf.reverse(self.x_eig1, axis=[2])[:, :, 0:l1]
            self.top_x_eig1 = tf.transpose(tf.reshape(self.top_x_eig1, [info.N_CHANNELS, k1, k2, l1]), [2, 1, 0, 3])

            self.filt1_viz = tf.transpose(self.top_x_eig1, [3, 0, 1, 2])
            tf.summary.image('filt1', self.filt1_viz, max_outputs=l1)

        with tf.name_scope("convolution1"):
            self.conv1 = tf.nn.conv2d(self._image_batch, self.top_x_eig1, strides=[1, 1, 1, 1], padding='SAME')
            self.conv1 = tf.transpose(self.conv1, [3, 0, 1, 2])
            # conv1 is now (l1, batch_size, img_w, img_h)
            self.conv1_batch = tf.expand_dims(tf.reshape(self.conv1, [-1, info.IMAGE_W, info.IMAGE_H]), axis=3)
            # conv1 batch is (l1 * batch_size, img_w, img_h)

            tf.summary.image('conv1', self.conv1_batch, max_outputs=l1)

        with tf.name_scope("extract_patches2"):
            self.patches2 = tf.extract_image_patches(self.conv1_batch, [1, k1, k2, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME', name='patches')
            self.patches2 = tf.expand_dims(tf.reshape(self.patches2, [-1, k1 * k2], name='patches_shaped'), axis=2)
            self.zero_mean_patches2 = self.patches2 - tf.reduce_mean(self.patches2, axis=1, keep_dims=True, name='patch_means')
            x2 = tf.transpose(self.zero_mean_patches2, [2, 1, 0])
            x2_trans = tf.transpose(self.zero_mean_patches2, [2, 0, 1])
            self.patches_covariance2 = tf.matmul(x2, x2_trans, name='patch_covariance')

        with tf.name_scope("eignvalue_decomposition2"):
            self.x_eig_vals2, self.x_eig2 = tf.self_adjoint_eig(self.patches_covariance2, name='x_eig')
            self.top_x_eig2 = tf.reverse(-self.x_eig2, axis=[2])[:, :, 0:l2]
            # negative sign makes it behave like MATLAB's eig, although the math is correct either way
            self.top_x_eig2 = tf.transpose(tf.reshape(self.top_x_eig2, [1, k1, k2, l2]), [2, 1, 0, 3])

            self.filt2_viz = tf.transpose(self.top_x_eig2, [3, 0, 1, 2])
            tf.summary.image('filt2', self.filt2_viz, max_outputs=l2)

        with tf.name_scope("convolution2"):
            self.conv2 = tf.nn.conv2d(self.conv1_batch, self.top_x_eig2, strides=[1, 1, 1, 1], padding='SAME')
            self.conv2 = tf.reshape(self.conv2, [l1, -1, info.IMAGE_W, info.IMAGE_H, l2])
            self.conv2 = tf.transpose(self.conv2, [0, 4, 1, 2, 3])
            # conv2 is now (l1, l2, batch_size, img_w, img_h)
            self.conv2_batch = tf.reshape(self.conv2, [-1, info.IMAGE_W, info.IMAGE_H, 1])

            tf.summary.image('conv2', self.conv2_batch, max_outputs=l2)

        with tf.name_scope("binary_quantize"):
            self.binary_quantize = tf.cast(self.conv2 > 0, tf.float32)
            self.powers_of_two = tf.constant([2 ** n for n in range(0, l2)], dtype=tf.float32)
            self.binary_encoded = tf.reduce_sum(tf.transpose(self.binary_quantize, [0, 2, 3, 4, 1]) * self.powers_of_two, axis=4)

            self.binary_quantize_viz = tf.reshape(tf.expand_dims(self.binary_quantize, axis=4), [-1, info.IMAGE_W, info.IMAGE_H, 1])
            self.binary_encoded_viz = tf.expand_dims(self.binary_encoded[:, 1, :, :], axis=3)
            tf.summary.image('quantized', self.binary_quantize_viz, max_outputs=10)

        with tf.name_scope("histograms"):
            self.n_bins = k = pow(2, l2)
            self.bins = np.linspace(-0.5, k - 0.5, self.n_bins)
            self.binary_flat = tf.expand_dims(tf.reshape(self.binary_encoded, [-1, info.IMAGE_W, info.IMAGE_H]), axis=3)
            self.blocks = tf.extract_image_patches(self.binary_flat, [1, block_w, block_h, 1], [1, stride_w, stride_h, 1], [1, 1, 1, 1], padding='VALID')
            self.blocks_flat = tf.reshape(self.blocks, [-1, block_w * block_h])
            self.blocks_flat_T = tf.transpose(self.blocks_flat, [1, 0])
            total_number_of_histograms = info.batch_size * l1 * num_blocks
            self.segment_ids = self.blocks_flat_T + [num_hist_bins * i for i in range(total_number_of_histograms)]
            self.segment_ids = tf.cast(tf.transpose(self.segment_ids, [1, 0]), tf.int32)
            number_of_segments = total_number_of_histograms * num_hist_bins
            self.histograms = tf.unsorted_segment_sum(tf.ones_like(self.blocks_flat), self.segment_ids, number_of_segments)
            self.histograms = tf.reshape(self.histograms, [l1, -1, num_blocks * num_hist_bins])

        self.output_features = tf.reshape(tf.transpose(self.histograms, [1, 0, 2]), [-1, l1 * num_blocks * num_hist_bins])

    def set_input_tensor(self, image_batch):
        self._image_batch = image_batch


def main():
    day_str = "{:%B_%d}".format(datetime.now())
    time_str = "{:%H_%MN%S}".format(datetime.now())
    day_dir = "log_data/" + day_str + "/"
    log_path = day_dir + day_str + "_" + time_str + "/"
    writer = tf.summary.FileWriter(log_path)
    if not os.path.exists(day_dir) and '--no-log' not in sys.argv:
        os.mkdir(day_dir)

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
    init = tf.global_variables_initializer()

    # Hyper-params
    k1 = 7
    k2 = 7
    l1 = 8
    l2 = 8
    block_w = 7
    block_h = 7
    block_overlap = 0.5
    num_hist_bins = 2 ** l2
    stride_w = max(int((1 - block_overlap) * block_w), 1)
    stride_h = max(int((1 - block_overlap) * block_h), 1)
    w_steps = range(block_w, info.IMAGE_W + 1, stride_w)
    h_steps = range(block_h, info.IMAGE_H + 1, stride_h)
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
    svm = LinearSVC(C=1, fit_intercept=False)
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
