#!/usr/bin/python3.5

# This code attempts to replicate some of these results:
# https://arxiv.org/pdf/1404.3606.pdf

import tensorflow as tf
from datetime import datetime
import os
import sys
from subprocess import call


# Global constants describing the CIFAR-10 data set.
N_CHANNELS = 3
IMAGE_SIZE = 32
img_dim = IMAGE_SIZE * IMAGE_SIZE * N_CHANNELS
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000


class PCANet:
    def __init__(self, image_batch, label_batch, global_step):
        self.image_batch = image_batch
        self.label_batch = label_batch

        k1 = 5
        k2 = 5
        l1 = 10
        self.patches = tf.extract_image_patches(image_batch, ksizes=[1, k1, k2, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME', name='x')
        self.patches = tf.reshape(self.patches, [-1, IMAGE_SIZE, IMAGE_SIZE, k1 * k2, N_CHANNELS])
        print(self.patches.get_shape())
        self.X = self.patches - tf.reduce_mean(self.patches, axis=3, keep_dims=True)
        # self.X_cov = tf.matmul(tf.expand_dims(self.X, axis=4), tf.expand_dims(self.X, axis=3), name='x_covariance')
        # _, x_eig = tf.self_adjoint_eig(self.X_cov, name='x_eig')
        # self.top_x_eig = tf.reshape(x_eig[0:l1], [-1, IMAGE_SIZE, IMAGE_SIZE, k1, k2, 3])
        # print(self.top_x_eig.get_shape())


def generate_image_and_label_batch(images, labels, min_queue_examples, batch_size):
    with tf.name_scope("make_batches"):
        num_preprocess_threads = 2
        images, label_batch = tf.train.shuffle_batch(
            [images, labels],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)

        tf.summary.image('images', images)

        return images, tf.one_hot(tf.squeeze(label_batch), depth=NUM_CLASSES)


def input_queue(filenames):
    queue = tf.train.string_input_producer(filenames)
    raw_img_dim = IMAGE_SIZE * IMAGE_SIZE * 3

    with tf.name_scope("read"):
        reader = tf.FixedLengthRecordReader(record_bytes=raw_img_dim + 1, name='input_reader')
        _, record_str = reader.read(queue, name='read_op')
        record_raw = tf.decode_raw(record_str, tf.uint8, name='decode_raw')

        label = tf.cast(tf.slice(record_raw, [0], [1]), tf.int32)
        image = tf.reshape(tf.slice(record_raw, [1], [raw_img_dim]), [3, 32, 32])
        float_image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)

    with tf.name_scope('preprocess'):
        norm_image = tf.divide(float_image, 255.0, name='norm_images')
        # gray_image = tf.reduce_sum(tf.multiply(norm_image, tf.constant([0.2126, 0.7512, 0.0722])), axis=2, name='grayscale')
        # gray_image = tf.expand_dims(gray_image, axis=2)
        # final_image = gray_image
        final_image = norm_image

    return final_image, label


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
    data_dir = 'cifar'
    train_filenames = [os.path.join(data_dir, 'data_batch_%i.bin' % i) for i in range(1, 6)]
    test_filenames = [os.path.join(data_dir, 'test_batch.bin')]

    train_images, train_labels = input_queue(train_filenames)
    test_images, test_labels = input_queue(test_filenames)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    train_image_batch, train_label_batch = generate_image_and_label_batch(train_images, train_labels, min_queue_examples, 128)
    test_image_batch, test_label_batch = generate_image_and_label_batch(test_images, test_labels, min_queue_examples, 10000)

    sess = tf.Session()

    tf.train.start_queue_runners(sess=sess)

    global_step = tf.Variable(0, trainable=False, name='global_step')

    # define the model
    m = PCANet(train_image_batch, train_label_batch, global_step)

    imgs, x, l = sess.run([m.image_batch, m.patches, m.label_batch])
    first_patch = x[0, 0, 0]
    patch = first_patch.reshape((5, 5, 3))
    img = imgs[0].reshape((IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS))
    print(l[0])
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(patch, interpolation='none')
    plt.figure()
    plt.imshow(img, interpolation='none')
    plt.show()


if __name__ == '__main__':
    main()
