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

        k1 = 5
        k2 = 5
        l1 = 25
        self.patches = tf.extract_image_patches(image_batch, ksizes=[1, k1, k2, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME', name='patches')
        self.patches = tf.reshape(self.patches, [-1, k1 * k2, info.N_CHANNELS], name='patches_shaped')
        self.zero_mean_patches = self.patches - tf.reduce_mean(self.patches, axis=1, keep_dims=True, name='patch_means')
        x = tf.transpose(self.zero_mean_patches, [2, 1, 0])
        x_trans = tf.transpose(self.zero_mean_patches, [2, 0, 1])
        self.patches_covariance = tf.matmul(x, x_trans, name='patch_covariance')
        self.x_eig_vals, self.x_eig = tf.self_adjoint_eig(self.patches_covariance, name='x_eig')
        self.x_eig = tf.squeeze(self.x_eig)
        self.top_x_eig = self.x_eig[:, 0:l1]
        self.top_x_eig = tf.transpose(tf.reshape(self.top_x_eig, [info.N_CHANNELS, k1, k2, l1]), [2, 1, 0, 3])

        self.conv1 = tf.nn.conv2d(image_batch, self.top_x_eig, strides=[1, 1, 1, 1], padding='SAME')
        self.conv1_viz = tf.reshape(tf.transpose(self.conv1, [0, 3, 1, 2]), [-1, info.IMAGE_W, info.IMAGE_H, info.N_CHANNELS])
        self.filt1_viz = tf.transpose(self.top_x_eig, [3, 0, 1, 2])
        self.patches_viz = tf.reshape(self.patches, [-1, 5, 5, info.N_CHANNELS])
        self.mean_patches_viz = tf.reshape(self.zero_mean_patches, [-1, 5, 5, info.N_CHANNELS])

        tf.summary.image('input', self.image_batch, max_outputs=10)
        tf.summary.image('conv1', self.conv1_viz, max_outputs=10)
        tf.summary.image('filt1', self.filt1_viz, max_outputs=l1)


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
        cmd = ['git', 'commit', '*.py']
        os.environ['TF_LOG_DIR'] = log_path
        call(cmd)

    # setup the input data pipelines
    train_image_batch, train_label_batch, test_image_batch, test_label_batch, info = load('cifar')

    sess = tf.Session()

    tf.train.start_queue_runners(sess=sess)
    init = tf.global_variables_initializer()

    # define the model
    m = PCANet(train_image_batch, train_label_batch, info)


    # run it
    sess.run(init)
    writer.add_graph(sess.graph)
    merged_summary = tf.summary.merge_all()

    _, summary = sess.run([m.conv1, merged_summary])
    writer.add_summary(summary, 0)

    writer.close()


if __name__ == '__main__':
    main()
