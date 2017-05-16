#!/usr/bin/python3.5

import numpy as np
import tensorflow as tf
from dataset_utils import MNIST


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def load_and_write(images_filename, labels_filename, writer):
    info = MNIST()
    images = open(images_filename, 'rb')
    labels = open(labels_filename, 'rb')

    # read and ignore header. we know the files are unsigned 8bit ints
    images.seek(16)
    labels.seek(8)

    np_images = np.fromfile(images, dtype=np.uint8).reshape((-1, info.img_dim()))
    np_labels = np.fromfile(labels, dtype=np.uint8)

    for img, label in zip(np_images, np_labels):
        image_bytes = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(info.IMAGE_H),
            'width': _int64_feature(info.IMAGE_W),
            'depth': _int64_feature(info.N_CHANNELS),
            'image': _bytes_feature(image_bytes),
            'label': _float_feature(label),
        }))
        writer.write(example.SerializeToString())


def main():
    info = MNIST()

    train_writer = tf.python_io.TFRecordWriter(info.TRAIN_RECORD_PATH)
    load_and_write('mnist/train_images', 'mnist/train_labels', train_writer)
    train_writer.close()

    test_writer = tf.python_io.TFRecordWriter(info.TEST_RECORD_PATH)
    load_and_write('mnist/test_images', 'mnist/test_labels', test_writer)
    test_writer.close()


if __name__ == '__main__':
    main()
