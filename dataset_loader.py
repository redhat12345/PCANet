#!/usr/bin/python3.5

import tensorflow as tf
import os

class DatasetInfo():
    def __init__(self):
        self.N_CHANNELS = 0
        self.IMAGE_W = None
        self.IMAGE_H = None
        self.NUM_CLASSES = 0

    def img_dim(self):
        return self.IMAGE_H * self.IMAGE_W * self.N_CHANNELS


def generate_image_and_label_batch(images, labels, min_queue_examples, batch_size, info):
    with tf.name_scope("make_batches"):
        num_preprocess_threads = 2
        images, label_batch = tf.train.shuffle_batch(
            [images, labels],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)

        tf.summary.image('images', images)

        return images, tf.one_hot(tf.squeeze(label_batch), depth=info.NUM_CLASSES)


def input_queue(filenames, info):
    queue = tf.train.string_input_producer(filenames)
    raw_img_dim = info.img_dim()

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


def load_cifar10():
    data_dir = 'cifar'
    train_filenames = [os.path.join(data_dir, 'data_batch_%i.bin' % i) for i in range(1, 6)]
    test_filenames = [os.path.join(data_dir, 'test_batch.bin')]

    # constants describing the CIFAR-10 data set.
    info = DatasetInfo()
    info.N_CHANNELS = 3
    info.IMAGE_H = 32
    info.IMAGE_W = 32
    info.NUM_CLASSES = 10

    train_images, train_labels = input_queue(train_filenames, info)
    test_images, test_labels = input_queue(test_filenames, info)

    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    train_image_batch, train_label_batch = generate_image_and_label_batch(train_images, train_labels, min_queue_examples, 128, info)
    test_image_batch, test_label_batch = generate_image_and_label_batch(test_images, test_labels, min_queue_examples, 10000, info)

    return train_image_batch, train_label_batch, test_image_batch, test_label_batch, info
