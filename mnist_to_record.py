#!/usr/bin/python3.5

import tensorflow as tf
import numpy as np
import os
import pickle
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

    images_header = images.read(4)
    labels_header = labels.read(4)

    print(images_header, labels_header)
    # dataset = pickle.load(fo, encoding='bytes')
    # images = dataset[bytes('data', encoding='utf-8')]
    # labels = dataset[bytes('labels', encoding='utf-8')]
    # for image, label in zip(images, labels):
    #     image = image.reshape(32, 32, 1)
    #     image_bytes = image.tobytes()
    #     example = tf.train.Example(features=tf.train.Features(feature={
    #         'height': _int64_feature(info.IMAGE_H),
    #         'width': _int64_feature(info.IMAGE_W),
    #         'depth': _int64_feature(info.N_CHANNELS),
    #         'image': _bytes_feature(image_bytes),
    #         'label': _float_feature(label),
    #     }))
    #     writer.write(example.SerializeToString())


def main():
    data_dir = 'mnist'
    info = MNIST()

    train_writer = tf.python_io.TFRecordWriter(info.TRAIN_RECORD_PATH)
    load_and_write('train_images', 'train_labels', train_writer)
    train_writer.close()

    test_writer = tf.python_io.TFRecordWriter(info.TEST_RECORD_PATH)
    load_and_write('train_images', 'train_labels', train_writer)
    test_writer.close()


if __name__ == '__main__':
    main()
