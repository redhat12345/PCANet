#!/usr/bin/python3.5

# This code attempts to replicate some of these results:
# https://arxiv.org/pdf/1404.3606.pdf

import tensorflow as tf
from datetime import datetime
from dataset_utils import load
import os
import sys
from subprocess import call


class PCANet:
    def __init__(self, image_batch, label_batch, info):
        self.image_batch = image_batch
        self.label_batch = label_batch

        tf.summary.image('input', self.image_batch, max_outputs=10)

        k1 = 5
        k2 = 5
        l1 = 12
        l2 = 8
        with tf.name_scope("extract_patches1"):
            self.patches1 = tf.extract_image_patches(image_batch, [1, k1, k2, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME', name='patches')
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
            self.conv1 = tf.nn.conv2d(image_batch, self.top_x_eig1, strides=[1, 1, 1, 1], padding='SAME')
            self.conv1 = tf.transpose(self.conv1, [0, 3, 1, 2])
            self.conv1_batch = tf.reshape(self.conv1, [-1, info.IMAGE_W, info.IMAGE_H, 1])

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
            self.top_x_eig2 = tf.reverse(self.x_eig2, axis=[2])[:, :, 0:l2]
            self.top_x_eig2 = tf.transpose(tf.reshape(self.top_x_eig2, [1, k1, k2, l2]), [2, 1, 0, 3])

            self.filt2_viz = tf.transpose(self.top_x_eig2, [3, 0, 1, 2])
            tf.summary.image('filt2', self.filt2_viz, max_outputs=l2)

        with tf.name_scope("convolution2"):
            self.conv2 = tf.nn.conv2d(self.conv1_batch, self.top_x_eig2, strides=[1, 1, 1, 1], padding='SAME')
            self.conv2 = tf.reshape(tf.transpose(self.conv2, [0, 3, 1, 2]), [-1, l1, l2, info.IMAGE_W, info.IMAGE_H])
            self.conv2_batch = tf.reshape(self.conv2, [-1, info.IMAGE_W, info.IMAGE_H, 1])
            tf.summary.image('conv2', self.conv2_batch, max_outputs=l2)

        with tf.name_scope("binary_quantize"):
            self.binary_quantize = tf.cast(self.conv2 > 0, tf.float32)
            self.powers_of_two = tf.constant([2 ** n for n in range(0, l2)], dtype=tf.float32)
            self.binary_encoded = tf.reduce_sum(tf.transpose(self.binary_quantize, [0, 1, 3, 4, 2]) * self.powers_of_two, axis=4)
            print(self.binary_encoded.get_shape())

            self.binary_quantize_viz = tf.reshape(tf.expand_dims(self.binary_quantize, axis=4), [-1, info.IMAGE_W, info.IMAGE_H, 1])
            self.binary_encoded_viz_1 = tf.expand_dims(self.binary_encoded[:, 1, :, :], axis=3)
            self.binary_encoded_viz_2 = tf.expand_dims(self.binary_encoded[:, 3, :, :], axis=3)
            tf.summary.image('quantized', self.binary_quantize_viz, max_outputs=l2)
            tf.summary.image('encoded', self.binary_encoded_viz_1, max_outputs=l2)
            tf.summary.image('encoded_2', self.binary_encoded_viz_2, max_outputs=l2)

        with tf.name_scope("histograms"):
            pass



def main():
    day_str = "{:%B_%d}".format(datetime.now())
    time_str = "{:%H:%M:%S}".format(datetime.now())
    day_dir = "log_data/" + day_str + "/"
    log_path = day_dir + day_str + "_" + time_str + "/"
    writer = tf.summary.FileWriter(log_path)
    if not os.path.exists(day_dir):
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

    # define the model
    m = PCANet(train_image_batch, train_label_batch, info)

    # run it
    sess.run(init)
    writer.add_graph(sess.graph)
    merged_summary = tf.summary.merge_all()

    _, summary, p = sess.run([m.conv2_batch, merged_summary, m.powers_of_two])
    writer.add_summary(summary, 0)

    writer.close()


if __name__ == '__main__':
    main()
