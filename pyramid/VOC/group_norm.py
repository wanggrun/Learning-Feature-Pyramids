#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: group_norm_conv.py
# Author: Guangrun Wang, Jiefeng Peng


import tensorflow as tf
from tensorflow.contrib.framework import add_model_variable
from tensorflow.python.training import moving_averages

from tensorpack.utils import logger
from tensorpack.tfutils.tower import get_current_tower_context
from tensorpack.tfutils.common import get_tf_version_number
from tensorpack.tfutils.collection import backup_collection, restore_collection
from tensorpack.models.common import layer_register, VariableHolder

__all__ = ['GroupNorm']

# decay: being too close to 1 leads to slow start-up. torch use 0.9.
# eps: torch: 1e-5. Lasagne: 1e-4


def get_bn_variables(n_out, use_scale, use_bias, gamma_init):
    if use_bias:
        beta = tf.get_variable('beta', [1, 1, 1, n_out], initializer=tf.constant_initializer())
    else:
        beta = tf.zeros([1, 1, 1, n_out], name='beta')
    if use_scale:
        gamma = tf.get_variable('gamma', [1, 1, 1, n_out], initializer=gamma_init)
    else:
        gamma = tf.ones([1, 1, 1, n_out], name='gamma')

    return beta, gamma


@layer_register()
def GroupNorm(inputs, G, training=None, momentum=0.9, epsilon=1e-5,
              center=True, scale=True,
              gamma_initializer=tf.ones_initializer(),
              data_format='channels_last',
              internal_update=False):
    """
    """
    data_format = 'channels_last'
    shape = inputs.get_shape().as_list()
    ndims = len(shape)
    assert ndims in [2, 4]
    if ndims == 2:
        data_format = 'channels_last'
    if data_format == 'channels_first':
        n_out = shape[1]
    else:
        n_out = shape[-1]  # channel
    assert n_out is not None, "Input to GroupNorm cannot have unknown channels!"
    beta, gamma = get_bn_variables(n_out, scale, center, gamma_initializer)

    if ndims == 2:
        inputs = tf.reshape(inputs, [-1, 1, 1, n_out])    # fused_bn only takes 4D input
         
    input_shape =inputs.get_shape().as_list()
    N = tf.shape(inputs)[0]
    C = input_shape[3]
    H = input_shape[1]
    W = input_shape[2]

    print(N, C, H, W, C//G)
    inputs = tf.reshape(inputs, [N, H, W, G, C//G])
    batch_mean, batch_var = tf.nn.moments(inputs, [1, 2, 4], keep_dims=True)
    xn = (inputs - batch_mean) / tf.sqrt(batch_var + epsilon)
    xn = tf.reshape(xn, [N, H, W, C])
    xn = xn * gamma + beta

    if ndims == 2:
        xn = tf.squeeze(xn, [1, 2])
   
    ret = tf.identity(xn, name='output')
    return ret
