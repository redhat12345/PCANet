import sys

import tensorflow as tf

from dataset_utils import load

def main():
    train_image_batch, train_label_batch, test_image_batch, test_label_batch, info = load('mnist')

    sess = tf.Session()

    tf.train.start_queue_runners(sess=sess)

    for i in range(10):
        images, labels = sess.run([train_image_batch, train_label_batch])
        tf.summary.image("image", images, max_outputs=10)

if __name__ == '__main__':
    main()
