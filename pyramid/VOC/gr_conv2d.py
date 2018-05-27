#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: gr_conv2d.py
# Author: Guangrun Wang <wanggrun@mail2.sysu.edu.cn>

import tensorflow as tf
from tensorpack.models.common import layer_register, VariableHolder
from tensorpack.models.tflayer import rename_get_variable
from tensorpack.utils.argtools import shape2d, shape4d

__all__ = ['GrConv2D']


@layer_register(log_shape=True)
def GrConv2D(x, out_channel, kernel_shape,
           padding='SAME', stride=1, dilation_rate=1,
           W_init=None, b_init=None,
           nl=tf.identity, split=1, use_bias=True,
           data_format='channels_last'):
    
    if data_format == 'NHWC' or data_format == 'channels_last':
        data_format = 'channels_last'
    elif data_format == 'NCHW' or data_format == 'channels_first':
        data_format = 'channels_first'
    else:
        print "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa unknown data format"
    in_shape = x.get_shape().as_list()
    channel_axis = 3 if data_format == 'NHWC' else 1
    in_channel = in_shape[channel_axis]
    assert in_channel is not None, "[GrConv2D] Input cannot have unknown channel!"
    assert in_channel % split == 0
    assert out_channel % split == 0

    kernel_shape = shape2d(kernel_shape)
    padding = padding.upper()
    filter_shape = kernel_shape + [in_channel / split, out_channel]
    stride = shape2d(stride)

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer()
    if b_init is None:
        b_init = tf.constant_initializer()


    with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
        layer = tf.layers.Conv2D(filters=out_channel, kernel_size=kernel_shape, strides=stride, 
            padding=padding, data_format=data_format,
            dilation_rate=dilation_rate, activation=lambda x: nl(x, name='output'), use_bias=use_bias,
            kernel_initializer=W_init, bias_initializer=b_init, trainable=True)            
        ret = layer.apply(x, scope=tf.get_variable_scope())

    ret.variables = VariableHolder(W=layer.kernel)
    if use_bias:
        ret.variables.b = layer.bias
    return ret

