#### Guangrun Wang <wanggrun@mail2.sysu.edu.cn>

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from tensorpack.tfutils import argscope
from tensorpack.models import (Conv2D, MaxPooling, GlobalAvgPooling, Dropout, BatchNorm)
from tensorpack.tfutils.tower import TowerContext

from coco_utils import (
    fbresnet_augmentor, get_voc_dataflow, MSC_Model,
    eval_on_ILSVRC12)
from resnet_model_voc_aspp import (
    preresnet_group, preresnet_basicblock, preresnet_bottleneck, 
    resnet_group, resnet_group_dilation,resnet_basicblock, resnet_bottleneck, resnet_bottleneck_dilation, se_resnet_bottleneck,
    resnet_backbone)

def load_variables_from_checkpoint(sess, start_checkpoint):
  """Utility function to centralize checkpoint restoration.

  Args:
    sess: TensorFlow session.
    start_checkpoint: Path to saved checkpoint on disk.
  """
  saver = tf.train.Saver(tf.global_variables())
  saver.restore(sess, start_checkpoint)

def create_resnet_model(image, is_training, isMSC=False, isASPP=False):
  mode = 'resnet'
  bottleneck = {
      'resnet': resnet_bottleneck,
      'preact': preresnet_bottleneck,
      'se': se_resnet_bottleneck}[mode]
  basicblock = preresnet_basicblock if mode == 'preact' else resnet_basicblock

  num_blocks, block_func = {
      18: ([2, 2, 2, 2], basicblock),
      34: ([3, 4, 6, 3], basicblock),
      50: ([3, 4, 6, 3], bottleneck),
      101: ([3, 4, 23, 3], bottleneck),
      152: ([3, 8, 36, 3], bottleneck)
  }[101]
  

  with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format='NHWC'):
    with argscope([BatchNorm], use_local_stat=False):
      logits = resnet_backbone(image, num_blocks, resnet_group, resnet_group_dilation, block_func, resnet_bottleneck_dilation)
      # image075 = tf.image.resize_images(image, [int(image.get_shape().as_list()[1]*0.75), int(image.get_shape().as_list()[2]*0.75)]) 
      # with tf.variable_scope('', reuse=True):
      #   logits075 = resnet_backbone(image075, num_blocks, resnet_group, resnet_group_dilation, block_func, resnet_bottleneck_dilation)
      # image05 = tf.image.resize_images(image, [int(image.get_shape().as_list()[1]*0.5), int(image.get_shape().as_list()[2]*0.5)]) 
      # with tf.variable_scope('', reuse=True):
      #   logits05 = resnet_backbone(image05, num_blocks, resnet_group, resnet_group_dilation, block_func, resnet_bottleneck_dilation)
      # logits = tf.reduce_max(tf.stack([logits100, tf.image.resize_images(logits075, tf.shape(logits100)[1:3, ]), tf.image.resize_images(logits05, tf.shape(logits100)[1:3, ])]), axis=0)
  # with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format='NHWC'):
  #   with argscope([BatchNorm], use_local_stat=False):
  #     logits = resnet_backbone(image, num_blocks, resnet_group, resnet_group_dilation, block_func, resnet_bottleneck_dilation)

  return logits
