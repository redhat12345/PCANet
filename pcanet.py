import tensorflow as tf

class PCANet:
    def __init__(self, init_image_batch, hyperparams, info):
        self._image_batch = init_image_batch
        tf.summary.image('input', self._image_batch, max_outputs=5)

        k1 = hyperparams['k1']
        k2 = hyperparams['k2']
        l1 = hyperparams['l1']
        l2 = hyperparams['l2']
        stride_w = hyperparams['stride_w']
        stride_h = hyperparams['stride_h']
        block_w = hyperparams['block_w']
        block_h = hyperparams['block_h']
        num_hist_bins = hyperparams['num_hist_bins']
        num_blocks = hyperparams['num_blocks']

        with tf.name_scope("extract_patches1"):
            self.patches1 = tf.extract_image_patches(self._image_batch, [1, k1, k2, 1], [1, 1, 1, 1], [1, 1, 1, 1], padding='SAME', name='patches')
            self.patches1 = tf.reshape(self.patches1, [-1, k1 * k2, info.N_CHANNELS], name='patches_shaped')
            # TODO: figure out how to unvectorize for multi-channel images
            # self.patches1 = tf.reshape(self.patches1, [-1, info.N_CHANNELS,  k1 * k2], name='patches_shaped')
            # self.patches1 = tf.transpose(self.patches1, [0, 2, 1])
            self.zero_mean_patches1 = self.patches1 - tf.reduce_mean(self.patches1, axis=1, keep_dims=True, name='patch_means')
            x1 = tf.transpose(self.zero_mean_patches1, [2, 1, 0])
            x1_trans = tf.transpose(self.zero_mean_patches1, [2, 0, 1])
            self.patches_covariance1 = tf.matmul(x1, x1_trans, name='patch_covariance')

        with tf.name_scope("eignvalue_decomposition1"):
            self.x_eig_vals1, self.x_eig1 = tf.self_adjoint_eig(self.patches_covariance1, name='x_eig')
            self.top_x_eig1 = tf.reverse(self.x_eig1, axis=[2])[:, :, 0:l1]
            self.top_x_eig1 = tf.transpose(tf.reshape(self.top_x_eig1, [info.N_CHANNELS, k1, k2, l1]), [2, 1, 0, 3])

            self.filt1_viz = tf.transpose(self.top_x_eig1, [3, 0, 1, 2])
            tf.summary.image('filt1', self.filt1_viz, max_outputs=l1)

        with tf.name_scope("convolution1"):
            self.conv1 = tf.nn.conv2d(self._image_batch, self.top_x_eig1, [1, 1, 1, 1], padding='SAME')
            self.conv1 = tf.transpose(self.conv1, [3, 0, 1, 2])
            # conv1 is now (l1, batch_size, img_w, img_h)
            self.conv1_batch = tf.expand_dims(tf.reshape(self.conv1, [-1, info.IMAGE_W, info.IMAGE_H]), axis=3)
            # conv1 batch is (l1 * batch_size, img_w, img_h)

            tf.summary.image('conv1', self.conv1_batch, max_outputs=l1)

        with tf.name_scope("extract_patches2"):
            self.patches2 = tf.extract_image_patches(self.conv1_batch, [1, k1, k2, 1], [1, 1, 1, 1], [1, 1, 1, 1], padding='SAME', name='patches')
            self.patches2 = tf.expand_dims(tf.reshape(self.patches2, [-1, k1 * k2], name='patches_shaped'), axis=2)
            self.zero_mean_patches2 = self.patches2 - tf.reduce_mean(self.patches2, axis=1, keep_dims=True, name='patch_means')
            x2 = tf.transpose(self.zero_mean_patches2, [2, 1, 0])
            x2_trans = tf.transpose(self.zero_mean_patches2, [2, 0, 1])
            self.patches_covariance2 = tf.matmul(x2, x2_trans, name='patch_covariance')

        with tf.name_scope("eignvalue_decomposition2"):
            self.x_eig_vals2, self.x_eig2 = tf.self_adjoint_eig(self.patches_covariance2, name='x_eig')
            self.top_x_eig2 = tf.reverse(-self.x_eig2, axis=[2])[:, :, 0:l2]
            # negative sign makes it behave like MATLAB's eig, although the math is correct either way
            self.top_x_eig2 = tf.transpose(tf.reshape(self.top_x_eig2, [1, k1, k2, l2]), [2, 1, 0, 3])

            self.filt2_viz = tf.transpose(self.top_x_eig2, [3, 0, 1, 2])
            tf.summary.image('filt2', self.filt2_viz, max_outputs=l2)

        with tf.name_scope("convolution2"):
            self.conv2 = tf.nn.conv2d(self.conv1_batch, self.top_x_eig2, [1, 1, 1, 1], padding='SAME')
            self.conv2 = tf.reshape(self.conv2, [l1, -1, info.IMAGE_W, info.IMAGE_H, l2])
            self.conv2 = tf.transpose(self.conv2, [0, 4, 1, 2, 3])
            # conv2 is now (l1, l2, batch_size, img_w, img_h)
            self.conv2_batch = tf.reshape(self.conv2, [-1, info.IMAGE_W, info.IMAGE_H, 1])

            tf.summary.image('conv2', self.conv2_batch, max_outputs=l2)

        with tf.name_scope("binary_quantize"):
            self.binary_quantize = tf.cast(self.conv2 > 0, tf.float32)
            self.powers_of_two = tf.constant([2 ** n for n in range(0, l2)], dtype=tf.float32)
            self.binary_encoded = tf.reduce_sum(tf.transpose(self.binary_quantize, [0, 2, 3, 4, 1]) * self.powers_of_two, axis=4)
            # binary_encoded is (l1, batch_size, img_w, img_h)

            self.binary_quantize_viz = tf.reshape(tf.expand_dims(self.binary_quantize, axis=4), [-1, info.IMAGE_W, info.IMAGE_H, 1])
            self.binary_encoded_viz = tf.expand_dims(self.binary_encoded[:, 1, :, :], axis=3)
            tf.summary.image('quantized', self.binary_quantize_viz, max_outputs=10)

        with tf.name_scope("histograms"):
            self.binary_flat = tf.expand_dims(tf.reshape(self.binary_encoded, [-1, info.IMAGE_W, info.IMAGE_H]), axis=3)
            self.blocks = tf.extract_image_patches(self.binary_flat, [1, block_w, block_h, 1], [1, stride_w, stride_h, 1], [1, 1, 1, 1], padding='VALID', name='blocks')
            # blocks is (l1*batch_size, len(w_steps), len(h_steps), block_w * block_h), ordered by l1
            # so the first batch_size are from the first filter in l1
            # values in blocks are in the range [0, 2^l2 - 1)
            self.blocks_flat = tf.reshape(self.blocks, [-1, block_w * block_h], name='blocks_flat')
            self.blocks_flat_T = tf.cast(tf.transpose(self.blocks_flat, [1, 0]), tf.int32, name='blocks_flat_T')
            total_number_of_histograms = info.batch_size * l1 * num_blocks

            # in order to histogram all the blocks in each image for each l1 filter
            # we construct a matrix of segment ids, then sum all elements in blocks with the same segment id
            # the offsets makes sure all values are unique across each image and l1 filter
            self.offsets = tf.convert_to_tensor([num_hist_bins * i for i in range(total_number_of_histograms)], dtype=tf.int32, name='offsets')
            self.segment_ids = tf.add(self.blocks_flat_T, self.offsets, name="add")
            self.segment_ids = tf.transpose(self.segment_ids, [1, 0])
            number_of_segments = total_number_of_histograms * num_hist_bins
            self.histograms = tf.unsorted_segment_sum(tf.ones_like(self.blocks_flat), self.segment_ids, number_of_segments)
            self.histograms = tf.reshape(self.histograms, [l1, -1, num_blocks * num_hist_bins])

        self.output_features = tf.reshape(tf.transpose(self.histograms, [1, 0, 2]), [-1, l1 * num_blocks * num_hist_bins])

    def set_input_tensor(self, image_batch):
        self._image_batch = image_batch


