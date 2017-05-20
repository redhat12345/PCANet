import tensorflow as tf
import sys


class DatasetInfo:
    def __init__(self):
        self.N_CHANNELS = 0
        self.IMAGE_W = None
        self.IMAGE_H = None
        self.NUM_CLASSES = 0
        self.NAME = "Unnamed DataSet"
        self.TRAIN_RECORD_PATH = None
        self.TEST_RECORD_PATH = None

    def img_dim(self):
        return self.IMAGE_H * self.IMAGE_W * self.N_CHANNELS


class Cifar10(DatasetInfo):
    def __init__(self):
        super().__init__()
        self.name = "CIFAR10"
        self.N_CHANNELS = 3
        self.IMAGE_H = 32
        self.IMAGE_W = 32
        self.NUM_CLASSES = 10
        self.TEST_RECORD_PATH = 'cifar/test_cifar10.tfrecords'
        self.TRAIN_RECORD_PATH = 'cifar/train_cifar10.tfrecords'


class MNIST(DatasetInfo):
    def __init__(self):
        super().__init__()
        self.name = "MNIST"
        self.N_CHANNELS = 1
        self.IMAGE_H = 28
        self.IMAGE_W = 28
        self.NUM_CLASSES = 10
        self.TEST_RECORD_PATH = 'mnist/test_mnist.tfrecords'
        self.TRAIN_RECORD_PATH = 'mnist/train_mnist.tfrecords'


def load(name):
    datasets = {
        'cifar10': Cifar10(),
        'mnist': MNIST(),
    }

    if name not in datasets:
        print(name + " is not one of the supported datasets:", datasets)
        sys.exit(0)

    info = datasets[name]

    batch_size = 1000

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([info.TRAIN_RECORD_PATH], name='train_queue')
        image, label = read_and_decode(filename_queue, info)

        train_image_batch, train_label_batch = tf.train.batch([image, label],
                                                                      batch_size=batch_size,
                                                                      capacity=1000 + 3 * batch_size,
                                                                      # seed=0,
                                                                      # min_after_dequeue=1000
                                                                      )
        filename_queue = tf.train.string_input_producer([info.TEST_RECORD_PATH], name='test_queue')
        image, label = read_and_decode(filename_queue, info)

        test_image_batch, test_label_batch = tf.train.batch([image, label],
                                                                    batch_size=batch_size,
                                                                    capacity=1000 + 3 * batch_size,
                                                                    # seed=0,
                                                                    # min_after_dequeue=1000
                                                                    )

    info.batch_size = batch_size
    return train_image_batch, train_label_batch, test_image_batch, test_label_batch, info


def read_and_decode(filename_queue, info):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.float32),
        })

    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([info.img_dim()])
    image = tf.reshape(image, [info.IMAGE_W, info.IMAGE_H, info.N_CHANNELS])
    image = tf.divide(tf.cast(image, tf.float32), 255.0, name='norm_images')
    label = features['label']

    return image, label
