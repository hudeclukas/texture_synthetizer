import tensorflow as tf
from tensorflow.contrib import slim as slim

import summary_utils as ssu

class texture_generator_network:
    def loss(self):
        with tf.variable_scope('loss_function'):
            diff = self.generator_nn - self.originals
            power = tf.pow(diff, 2)
            flat = slim.flatten(power)
            reduced = tf.reduce_sum(flat)
            losses = tf.sqrt(reduced)
            loss = tf.reduce_mean(losses)

            tf.summary.histogram('losses_batch',losses)
            tf.summary.scalar('loss_reduced',loss)
            return loss

    def architecture(self,input_tensor:tf.Tensor, name:str):
        ssu.image_summary(input_tensor,(150,150,1),0,'input_image')
        conv1 = slim.conv2d(
            inputs=input_tensor,
            num_outputs=96,
            kernel_size=[5, 5],
            # stride=[2,2],
            reuse=tf.AUTO_REUSE,
            activation_fn=tf.nn.sigmoid,
            scope= name + '_conv_1'
        )
        sliced = tf.slice(conv1, (0, 0, 0, 0), (1, -1, -1, 1))
        tf.summary.image('gnn_conv1_ss', sliced, max_outputs=1)
        conv2 = slim.conv2d(
            inputs=conv1,
            num_outputs=128,
            kernel_size=[5, 5],
            # stride=[3, 3],
            reuse=tf.AUTO_REUSE,
            activation_fn=tf.nn.sigmoid,
            scope=name + '_conv_2'
        )
        dropout1 = slim.dropout(
            conv2,
            scope=name + '_dropout_1',
            keep_prob=self.dropout_keep_prob,
        )
        conv3 = slim.conv2d(
            inputs=dropout1,
            num_outputs=128,
            kernel_size=[5, 5],
            reuse=tf.AUTO_REUSE,
            activation_fn=tf.nn.sigmoid,
            scope=name + '_conv_3'
        )
        batch_norm1 = slim.batch_norm(
            inputs=conv3,
            scope=name+'_batch_norm_1'
        )
        conv4_middle = slim.conv2d(
            inputs=batch_norm1,
            num_outputs=1,
            kernel_size=[5, 5],
            # stride=1,
            reuse=tf.AUTO_REUSE,
            activation_fn=tf.nn.sigmoid,
            scope=name + '_conv_4_middle'
        )
        ssu.image_summary(conv4_middle, (150, 150, 1), 0, 'gnn_output')
        return conv4_middle

        # dropout2 = slim.dropout(
        #     conv4_middle,
        #     scope=name + '_dropout_2',
        #     keep_prob=self.dropout_keep_prob,
        # )
        # conv5 = slim. conv2d(
        #     inputs=dropout2,
        #     num_outputs=128,
        #     kernel_size=[5, 5],
        #     # stride=[2, 2],
        #     reuse=tf.AUTO_REUSE,
        #     scope=name + '_conv_5'
        # )
        # conv6 = slim.conv2d(
        #     inputs=conv5,
        #     num_outputs=64,
        #     kernel_size=[5, 5],
        #     # stride=[2, 2],
        #     reuse=tf.AUTO_REUSE,
        #     scope=name + '_conv_6'
        # )
        # dropout3 = slim.dropout(
        #     conv6,
        #     scope=name + '_dropout_3',
        #     keep_prob=self.dropout_keep_prob,
        # )
        # batch_norm2 = slim.batch_norm(
        #     inputs=dropout3,
        #     scope=name + '_batch_norm_2'
        # )
        # conv7 = slim.conv2d(
        #     inputs=batch_norm2,
        #     num_outputs=1,
        #     kernel_size=[5, 5],
        #     # stride=[2, 2],
        #     reuse=tf.AUTO_REUSE,
        #     scope=name + '_conv_7'
        # )
        #
        # ssu.image_summary(conv7,(150,150,1),0,'gnn_output')
        # return conv7

    def __init__(self, image_size: tuple):
        with tf.variable_scope('generator'):
            self.originals = tf.placeholder(tf.float32, [None, 150, 150, 1], name="originals")
            self.input_batch = tf.placeholder(dtype=tf.float32, shape=[None, image_size[0], image_size[1], image_size[2]],
                                                name="input_batch")
            self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='droput_keep_prob')

            self.generator_nn = self.architecture(self.input_batch, 'gnn')

            self.loss = self.loss()