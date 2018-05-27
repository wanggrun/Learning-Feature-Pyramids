#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resnet_model.py

import tensorflow as tf


from tensorpack.tfutils.argscope import argscope, get_arg_scope
from tensorpack.models import (
    Conv2D, GlobalAvgPooling, BatchNorm, BNReLU, FullyConnected,
    LinearWrap)
from gr_resize import Grresize
from gr_conv2d import GrConv2D
# from tensorpack.tfutils.varreplace import freeze_variable
from tensorpack.tfutils.varreplace import freeze_variables


def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.get_shape().as_list()[1 if data_format in ['NCHW', 'channels_first'] else 3]
    if n_in != n_out:   # change dimension when channel is not the same
        return Conv2D('convshortcut', l, n_out, 1, strides=stride, activation=activation)
    else:
        return l


def apply_preactivation(l, preact):
    if preact == 'bnrelu':
        shortcut = l    # preserve identity mapping
        l = BNReLU('preact', l)
    else:
        shortcut = l
    return l, shortcut


def get_bn(zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    if zero_init:
        return lambda x, name=None: BatchNorm('bn', x, gamma_initializer=tf.zeros_initializer())
    else:
        return lambda x, name=None: BatchNorm('bn', x)


def preresnet_basicblock(l, ch_out, stride, preact):
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3)
    return l + resnet_shortcut(shortcut, ch_out, stride)


def preresnet_bottleneck(l, ch_out, stride, preact):
    # stride is applied on the second conv, following fb.resnet.torch
    l, shortcut = apply_preactivation(l, preact)
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride)


def preresnet_group(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                # first block doesn't need activation
                l = block_func(l, features,
                               stride if i == 0 else 1,
                               'no_preact' if i == 0 else 'bnrelu')
        # end of each group need an extra activation
        l = BNReLU('bnlast', l)
    return l


def resnet_basicblock(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, activation=get_bn(zero_init=True))
    return l + resnet_shortcut(shortcut, ch_out, stride, activation=get_bn(zero_init=False))


def resnet_bottleneck(l, ch_out, stride, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, strides=stride if stride_first else 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=1 if stride_first else stride, activation=BNReLU)   
    l = Conv2D('conv6', l, ch_out * 4, 1, activation=get_bn(zero_init=True))

    ldown =Conv2D('conv3', shortcut, ch_out / 8, 1, stride=stride if stride_first else 1, activation=BNReLU)
    
    hi = shortcut.get_shape().as_list()[1]
    wi = shortcut.get_shape().as_list()[2]
    ho = l.get_shape().as_list()[1]
    wo = l.get_shape().as_list()[2]

    lf = Grresize('grdownf', ldown, [ int(hi * 0.75), int(wi * 0.75)] )
    lf = Conv2D('conv2f', lf, ch_out / 8, 3, stride=1 if stride_first else stride, activation=BNReLU)
    lf = Grresize('grupsf', lf, [ho, wo])

    lg = Grresize('grdowng', ldown, [int(hi * 0.875), int(wi * 0.875)] )
    lg = Conv2D('conv2g', lg, ch_out / 8, 3, stride=1 if stride_first else stride, activation=BNReLU)
    lg = Grresize('grupsg', lg, [ho, wo])

    lh = Grresize('grdownh', ldown, [int(hi * 1.16), int(wi * 1.16) ] )
    lh = Conv2D('conv2h', lh, ch_out / 8, 3, stride=1 if stride_first else stride, activation=BNReLU)
    lh = Grresize('grupsh', lh, [ho, wo])

    li = Grresize('grdowni', ldown, [int(hi * 1.33), int(wi * 1.33)] )
    li = Conv2D('conv2i', li, ch_out / 8, 3, stride=1 if stride_first else stride, activation=BNReLU)
    li = Grresize('grupsi', li, [ho, wo])

    laug = tf.concat([lf, lg, lh, li], axis = 3)
    laug = Conv2D('conv6aug', laug, ch_out * 4, 1, activation=get_bn(zero_init=True))

    return l + laug + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))


def resnet_bottleneck_dilation(l, ch_out, stride, stride_first=False, dilation_rate = 1):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, strides=stride if stride_first else 1, activation=BNReLU)
    l = GrConv2D('conv2', l, ch_out, 3, stride=1 if stride_first else stride, dilation_rate=dilation_rate, nl=BNReLU)   
    l = Conv2D('conv6', l, ch_out * 4, 1, activation=get_bn(zero_init=True))

    ldown =Conv2D('conv3', shortcut, ch_out / 8, 1, stride=stride if stride_first else 1, activation=BNReLU)
    
    hi = shortcut.get_shape().as_list()[1]
    wi = shortcut.get_shape().as_list()[2]
    ho = l.get_shape().as_list()[1]
    wo = l.get_shape().as_list()[2]

    lf = Grresize('grdownf', ldown, [ int(hi * 0.75), int(wi * 0.75)] )
    lf = GrConv2D('conv2f', lf, ch_out / 8, 3, stride=1 if stride_first else stride, dilation_rate=dilation_rate, nl=BNReLU)
    lf = Grresize('grupsf', lf, [ho, wo])

    lg = Grresize('grdowng', ldown, [int(hi * 0.875), int(wi * 0.875)] )
    lg = GrConv2D('conv2g', lg, ch_out / 8, 3, stride=1 if stride_first else stride, dilation_rate=dilation_rate, nl=BNReLU)
    lg = Grresize('grupsg', lg, [ho, wo])

    lh = Grresize('grdownh', ldown, [int(hi * 1.16), int(wi * 1.16) ] )
    lh = GrConv2D('conv2h', lh, ch_out / 8, 3, stride=1 if stride_first else stride, dilation_rate=dilation_rate, nl=BNReLU)
    lh = Grresize('grupsh', lh, [ho, wo])

    li = Grresize('grdowni', ldown, [int(hi * 1.33), int(wi * 1.33)] )
    li = GrConv2D('conv2i', li, ch_out / 8, 3, stride=1 if stride_first else stride, dilation_rate=dilation_rate, nl=BNReLU)
    li = Grresize('grupsi', li, [ho, wo])

    laug = tf.concat([lf, lg, lh, li], axis = 3)
    laug = Conv2D('conv6aug', laug, ch_out * 4, 1, activation=get_bn(zero_init=True))

    return l + laug + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))


def se_resnet_bottleneck(l, ch_out, stride):
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, activation=BNReLU)
    l = Conv2D('conv2', l, ch_out, 3, strides=stride, activation=BNReLU)
    l = Conv2D('conv3', l, ch_out * 4, 1, activation=get_bn(zero_init=True))

    squeeze = GlobalAvgPooling('gap', l)
    squeeze = FullyConnected('fc1', squeeze, ch_out // 4, activation=tf.nn.relu)
    squeeze = FullyConnected('fc2', squeeze, ch_out * 4, activation=tf.nn.sigmoid)
    data_format = get_arg_scope()['Conv2D']['data_format']
    ch_ax = 1 if data_format in ['NCHW', 'channels_first'] else 3
    shape = [-1, 1, 1, 1]
    shape[ch_ax] = ch_out * 4
    l = l * tf.reshape(squeeze, shape)
    return l + resnet_shortcut(shortcut, ch_out * 4, stride, activation=get_bn(zero_init=False))


def resnet_group(l, name, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
                # end of each block need an activation
                l = tf.nn.relu(l)
    return l


def resnet_group_dilation(l, name, block_func_dilation, features, count, stride, dilation_rate):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func_dilation(l, features, stride if i == 0 else 1, dilation_rate = dilation_rate)
                # end of each block need an activation
                l = tf.nn.relu(l)
    return l


def resnet_backbone(image, num_blocks, group_func, group_func_dilation, block_func, block_func_dilation):
    with argscope(Conv2D, use_bias=False,
                  kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        # with freeze_variables(stop_gradient=True, skip_collection=True):
        logits = (LinearWrap(image)
                  .Conv2D('conv0', 64, 7, strides=2, activation=BNReLU)
                  .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                  .apply(group_func, 'group0', block_func, 64, num_blocks[0], 1)
                  .apply(group_func, 'group1', block_func, 128, num_blocks[1], 2)
                  .apply(group_func_dilation, 'group2', block_func_dilation, 256, num_blocks[2], 1, 2)
                  .apply(group_func_dilation, 'group3', block_func_dilation, 512, num_blocks[3], 1, 4))
        logits = (logits.Conv2D('conv102', 21, 1, stride=1, activation=tf.identity)())

        # logits = logits.Conv2D('conv102', 21, 1, stride=1, nl=tf.identity)()
        # tf.get_default_graph().clear_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # with tf.variable_scope('conv102', reuse=True):
        #     W = tf.get_variable('W')
        #     tf.add_to_collection(tf.GraphKeys.TRAINABLE_VARIABLES, W)
                
    return logits
