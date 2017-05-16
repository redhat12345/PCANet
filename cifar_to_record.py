#!/usr/bin/python3.5

import tensorflow as tf
import numpy as np
import os
import pickle
from dataset_utils import Cifar10


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def unpickle_and_write(filename, writer):
    info = Cifar10()
    with open(filename, 'rb') as fo:
        dataset = pickle.load(fo, encoding='bytes')
        images = dataset[bytes('data', encoding='utf-8')]
        labels = dataset[bytes('labels', encoding='utf-8')]
        for image, label in zip(images, labels):
            image = image.reshape(3, 32, 32)
            image = np.transpose(image, [1, 2, 0])
            image_bytes = image.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(info.IMAGE_H),
                'width': _int64_feature(info.IMAGE_W),
                'depth': _int64_feature(info.N_CHANNELS),
                'image': _bytes_feature(image_bytes),
                'label': _float_feature(label),
            }))
            writer.write(example.SerializeToString())


def main():
    data_dir = 'cifar'

    train_record_filename = os.path.join(data_dir, 'train_cifar10.tfrecords')
    train_writer = tf.python_io.TFRecordWriter(train_record_filename)
    for train_filename in [os.path.join(data_dir, 'data_batch_%i' % i) for i in range(1, 6)]:
        unpickle_and_write(train_filename, train_writer)
    train_writer.close()

    test_record_filename = os.path.join(data_dir, 'test_cifar10.tfrecords')
    test_writer = tf.python_io.TFRecordWriter(test_record_filename)
    unpickle_and_write(os.path.join(data_dir, 'test_batch'), test_writer)
    test_writer.close()


if __name__ == '__main__':
    main()
