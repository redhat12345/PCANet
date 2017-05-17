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
            self.conv1_batch = tf.reshape(tf.transpose(self.conv1, [0, 3, 1, 2]), [-1, info.IMAGE_W, info.IMAGE_H, 1])

            tf.summary.image('conv1', self.conv1_batch, max_outputs=l1)

        # with tf.name_scope("extract_patches2"):
        #     self.patches2 = tf.extract_image_patches(self.conv1_batch, [1, k1, k2, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME', name='patches')
        #     self.patches2 = tf.expand_dims(tf.reshape(self.patches2, [-1, k1 * k2], name='patches_shaped'), axis=2)
        #     self.zero_mean_patches2 = self.patches2 - tf.reduce_mean(self.patches2, axis=1, keep_dims=True, name='patch_means')
        #     x2 = tf.transpose(self.zero_mean_patches2, [2, 1, 0])
        #     x2_trans = tf.transpose(self.zero_mean_patches2, [2, 0, 1])
        #     self.patches_covariance2 = tf.matmul(x2, x2_trans, name='patch_covariance')
        #
        # with tf.name_scope("eignvalue_decomposition2"):
        #     self.x_eig_vals2, self.x_eig2 = tf.self_adjoint_eig(self.patches_covariance2, name='x_eig')
        #     self.top_x_eig2 = tf.reverse(self.x_eig2, axis=[2])[:, :, 0:l2]
        #     self.top_x_eig2 = tf.transpose(tf.reshape(self.top_x_eig2, [1, k1, k2, l2]), [2, 1, 0, 3])
        #
        #     self.filt2_viz = tf.transpose(self.top_x_eig2, [3, 0, 1, 2])
        #     tf.summary.image('filt2', self.filt2_viz, max_outputs=l2)

        # with tf.name_scope("convolution2"):
        #     self.conv2 = tf.nn.conv2d(self.conv1_batch, self.top_x_eig2, strides=[1, 1, 1, 1], padding='SAME')
        #     self.conv2 = tf.reshape(tf.transpose(self.conv2, [0, 3, 1, 2]), [-1, l1, l2, info.IMAGE_W, info.IMAGE_H])
        #     self.conv2_batch = tf.reshape(self.conv2, [-1, info.IMAGE_W, info.IMAGE_H, 1])
        #     tf.summary.image('conv2', self.conv2_batch, max_outputs=l2)

        with tf.name_scope("h1"):
            self.h1_dim = 128
            self.w_fc1 = tf.Variable(tf.truncated_normal([info.IMAGE_W * info.IMAGE_H * l1,  self.h1_dim], 0, 0.1), name='w_fc1')
            self.b_fc1 = tf.Variable(tf.constant(0.1, shape=[self.h1_dim]), name='b_fc1')
            self.conv1_flat = tf.reshape(self.conv1, [-1, info.IMAGE_W * info.IMAGE_H * l1])
            self.h1 = tf.nn.relu(self.conv1_flat @ self.w_fc1 + self.b_fc1, name='h1')

        with tf.name_scope("h2"):
            self.h2_dim = info.NUM_CLASSES
            self.w_fc2 = tf.Variable(tf.truncated_normal([self.h1_dim, self.h2_dim], 0, 0.1), name='w_fc2')
            self.b_fc2 = tf.Variable(tf.constant(0.1, shape=[self.h2_dim]), name='b_fc2')
            self.h2 = tf.nn.relu(self.h1 @ self.w_fc2 + self.b_fc2, name='h2')

        with tf.name_scope("train"):
            self.label_batch_one_hot = tf.one_hot(tf.cast(label_batch, tf.int32), depth=info.NUM_CLASSES)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.h2, labels=self.label_batch_one_hot), name='loss')
            self.train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
            self.predictions = tf.cast(tf.argmax(tf.nn.softmax(self.h2)), tf.float32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.label_batch, self.predictions), tf.float32), name='accuracy')

            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)


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

    # define the model
    m = PCANet(train_image_batch, train_label_batch, info)

    # run it
    init = tf.global_variables_initializer()
    sess.run(init)
    writer.add_graph(sess.graph)
    merged_summary = tf.summary.merge_all()

    for i in range(100):
        _, summary, l = sess.run([m.train, merged_summary, m.loss])
        print(l, i)
        writer.add_summary(summary, i)

    writer.close()


if __name__ == '__main__':
    main()
