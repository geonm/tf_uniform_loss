import os
import sys
import numpy as np
import tensorflow as tf

def uniform_loss(features, labels, num_class):
    centers = tf.get_variable(name='features_centers', shape=[class_num, features.get_shape().as_list()[-1]],
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(0),
                              trainable=False) 
    batch_size = features.get_shape().as_list()[0]

    labels = tf.cast(labels, tf.int32)

    selected_centers = tf.gather(centers, labels)

    diff = selected_centers - features

    labels_rh = tf.reshape(labels, [-1, 1])

    adjacency = tf.equal(labels_rh, tf.transpose(labels_rh))
    adjacency_not = tf.cast(tf.logical_not(adjacency), tf.float32)
    adjacency = tf.cast(adjacency, tf.float32)

    denom = tf.reduce_sum(adjacency, axis=1, keepdims=True)
    diff /= denom
    centers = tf.scatter_sum(centers, labels, diff)

    with tf.control_dependencies([centers]): # update centers first
        a = tf.reduce_sum(tf.square(selected_centers), axis=1, keepdims=True)
        b = tf.reduce_sum(tf.square(tf.transpose(selected_centers)), axis=0, keepdims=True)
        ab = tf.matmul(selected_centers, selected_centers, transpose_b=True)

        pd_mat = tf.add(a, b) - 2.0 * ab
        error_mask = tf.less_equal(pd_mat, 0.0)
        pd_mat = tf.sqrt(pd_mat + tf.to_float(error_mask) * 1e-16) + 1.0
        pd_mat = tf.multiply(1.0 / pd_mat, adjacency_not)

        uniform_loss = tf.reduce_sum(pd_mat) / (batch_size * (batch_size - 1.0))

    return uniform_loss
