#!/usr/bin/python3.5
from datetime import datetime
import os

import tensorflow as tf

from dataset_utils import load


def main():
    tag = "{:%B_%d_%H_%M_%S}".format(datetime.now())
    writer = tf.summary.FileWriter('log_data/test_dataset/' + tag)

    train_image_batch, train_label_batch, test_image_batch, test_label_batch, info = load('mnist')

    tf.summary.image("train_image", train_image_batch, max_outputs=10)
    tf.summary.image("test_image", test_image_batch, max_outputs=10)
    merged_summary = tf.summary.merge_all()

    sess = tf.Session()

    tf.train.start_queue_runners(sess=sess)

    for i in range(10):
        train_labels, test_labels, summary = sess.run([train_label_batch, test_label_batch, merged_summary])
        writer.add_summary(summary, i)
        print(train_labels[:10], test_labels[:10])

    writer.close()

if __name__ == '__main__':
    main()
