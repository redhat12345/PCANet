#!/usr/bin/python3.5

# This code attempts to replicate some of these results:
# https://arxiv.org/pdf/1404.3606.pdf

import tensorflow as tf
from datetime import datetime
from dataset_loader import load_cifar10, DatasetInfo
import os
import sys
from subprocess import call


class PCANet:
    def __init__(self, image_batch, label_batch, info):
        self.image_batch = image_batch
        self.label_batch = label_batch

        k1 = 5
        k2 = 5
        l1 = 10
        self.patches = tf.extract_image_patches(image_batch, ksizes=[1, k1, k2, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME', name='x')
        self.patches = tf.reshape(self.patches, [-1, k1 * k2, info.N_CHANNELS])
        self.zero_mean_patches = self.patches - tf.reduce_mean(self.patches, axis=2, keep_dims=True)
        x = tf.transpose(self.zero_mean_patches, [2, 1, 0])
        x_trans = tf.transpose(self.zero_mean_patches, [2, 0, 1])
        self.patches_covariance = tf.matmul(x, x_trans, name='patch_covariance')
        _, self.x_eig = tf.self_adjoint_eig(self.patches_covariance, name='x_eig')
        self.top_x_eig = tf.transpose(tf.reshape(self.x_eig[:, 0:l1], [3, k1, k2, l1]), [3, 2, 1, 0])


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
        cmd = ['git', 'commit', __file__]
        os.environ['TF_LOG_DIR'] = log_path
        call(cmd)

    # setup the input data pipelines
    train_image_batch, train_label_batch, test_image_batch, test_label_batch, info = load_cifar10()

    sess = tf.Session()

    tf.train.start_queue_runners(sess=sess)

    # define the model
    m = PCANet(train_image_batch, train_label_batch, info)

    e = sess.run(m.top_x_eig)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(e[0], interpolation='none')
    plt.show()


if __name__ == '__main__':
    main()
