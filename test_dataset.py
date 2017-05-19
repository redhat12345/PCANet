#!/usr/bin/python3.5
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from dataset_utils import load


class MyClass:
    def __init__(self, images):
        self._images = images


def main():
    tag = "{:%B_%d_%H_%M_%S}".format(datetime.now())
    writer = tf.summary.FileWriter('log_data/test_dataset/' + tag)

    train_image_batch, train_label_batch, test_image_batch, test_label_batch, info = load('mnist')

    tf.summary.image("train_image", train_image_batch, max_outputs=5)
    tf.summary.image("test_image", test_image_batch, max_outputs=5)

    sess = tf.Session()

    tf.train.start_queue_runners(sess=sess)

    m = MyClass(train_image_batch)
    m._images = test_image_batch

    merged_summary = tf.summary.merge_all()

    for i in range(10):
        train, test, img, summary = sess.run([train_label_batch, test_label_batch, m._images, merged_summary])
        plt.imshow(np.squeeze(img[0]), interpolation='none', cmap='gray')
        plt.show()
        print(train[:5], test[:5])
        writer.add_summary(summary, i)

    writer.close()

if __name__ == '__main__':
    main()
