from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import os.path
import sys

import pandas as pd
import datetime

import numpy as np
from six.moves import xrange  
import tensorflow as tf

import gr_models
from tensorflow.python.platform import gfile
from tensorpack.tfutils.tower import TowerContext

import pandas as pd
import cv2
from tqdm import tqdm
import pydensecrf.densecrf as dcrf

FLAGS = None
os.environ['CUDA_VISIBLE_DEVICES'] = "5"

CROPSHAPE = 480

def data_crop_test_output(session, gr_data, logits, image, mean, std, crop_size, stride):
    image_h = image.shape[0]
    image_w = image.shape[1]
    image = np.asarray(image, dtype='float32')
    image = image * (1.0 / 255)      
    image = (image - mean) / std
    pad_h = 0
    pad_w = 0
    if image_h >= crop_size and image_w >= crop_size:
        image_pad = image
    else:
        if image_h < crop_size:
            pad_h = crop_size - image_h
        if image_w < crop_size:
            pad_w = crop_size - image_w
        image_pad = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
    image_crop_batch = []
    x_start = range(0, image_pad.shape[0] - crop_size + 1, stride)
    y_start = range(0, image_pad.shape[1] - crop_size + 1, stride)
    if (image_pad.shape[0]-crop_size)%stride != 0:
        x_start.append(image_pad.shape[0]-crop_size)
    if (image_pad.shape[1]-crop_size)%stride != 0:
        y_start.append(image_pad.shape[1]-crop_size)
    for x in x_start:
        for y in y_start:
            image_crop_batch.append(image_pad[x:x+crop_size, y:y+crop_size])

    logits = session.run(
        logits,
        feed_dict={
            gr_data: image_crop_batch,
        })
    num_class = 21
    score_map = np.zeros([image_pad.shape[0], image_pad.shape[1], num_class], dtype = 'float32')
    count = np.zeros([image_pad.shape[0], image_pad.shape[1], num_class], dtype = 'float32')
    crop_index = 0
    for x in x_start:
        for y in y_start:
    	    crop_logits = logits[crop_index]
            score_map[x:x+crop_logits.shape[0], y:y+crop_logits.shape[1]] += crop_logits
            count[x:x+crop_logits.shape[0], y:y+crop_logits.shape[1]] += 1 
            crop_index += 1
     
    score_map = score_map[:image_h,:image_w] / count[:image_h,:image_w]
    return score_map

def dense_crf_batch(probs, img=None, n_iters=10, 
              sxy_gaussian=(1, 1), compat_gaussian=4,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(49, 49), compat_bilateral=5,
              srgb_bilateral=(13, 13, 13),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.
    
    Args:
      probs: class probabilities per pixel.
      img: if given, the pairwise bilateral potential on raw RGB values will be computed.
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      
    Returns:
      Refined predictions after MAP inference.
    """
    n, h, w, _ = probs.shape
    
    if img is not None:
        assert(img.shape[0:3] == (n, h, w)), "The image height and width must coincide with dimensions of the logits."
        for i in range(n):
            probs[i] = dense_crf(probs[i], img[i])
    else:
        for i in range(n):
            probs[i] = dense_crf(probs[i])
    return probs

def dense_crf(probs, img=None, n_iters=10, 
              sxy_gaussian=(1, 1), compat_gaussian=4,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(49, 49), compat_bilateral=5,
              srgb_bilateral=(13, 13, 13),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.
    
    Args:
      probs: class probabilities per pixel.
      img: if given, the pairwise bilateral potential on raw RGB values will be computed.
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      
    Returns:
      Refined predictions after MAP inference.
    """
    n_classes = 21
    h, w, _ = probs.shape
    
    probs = probs.transpose(2, 0, 1).copy(order='C') # Need a contiguous array.
    d = dcrf.DenseCRF2D(w, h, n_classes) # Define DenseCRF model.
    U = -np.log(probs) # Unary potential.
    U = U.reshape((n_classes, -1)) # Needs to be flat.
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=kernel_gaussian, normalization=normalisation_gaussian)
    if img is not None:
        assert(img.shape[0:2] == (h, w)), "The image height and width must coincide with dimensions of the logits."
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                               kernel=kernel_bilateral, normalization=normalisation_bilateral,
                               srgb=srgb_bilateral, rgbim=img)
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
    return preds


def randomCropPad(image, crop_h, crop_w):
    
    image_h = image.shape[0]
    image_w = image.shape[1]    
    pad_h = max(0, crop_h - image_h)
    pad_w = max(0, crop_w - image_w)
    image_pad = np.lib.pad(image, ((0,pad_h), (0,pad_w), (0,0)), 'constant', 
        constant_values=((128,128), (128,128), (128, 128)))
    comb_h = image_pad.shape[0]
    comb_w = image_pad.shape[1]

    crop_scale = 0.5

    crop_shift_h = int(crop_scale * float((comb_h - crop_h))/2)
    crop_shift_w = int(crop_scale * float((comb_w - crop_w))/2)

    image_crop = image_pad[crop_shift_h:crop_shift_h+crop_h, crop_shift_w:crop_shift_w+crop_w, :]    
    return image_crop



def main(_):

  batch_size = 32
  input_size = CROPSHAPE
  num_class = 21
  data_shape = CROPSHAPE
  CRF = True
  fname = []
  output = []
  #mean = [0.406, 0.485, 0.456]    
  #std = [0.225, 0.229, 0.224]
  # mean = [0.406, 0.456, 0.485]    
  # std = [0.225, 0.224, 0.229]

  mean = [0.406, 0.456, 0.485]
  std =  [0.225, 0.224, 0.229]

  mean = np.asarray(mean, dtype='float32')
  std = np.asarray(std, dtype='float32')
  crop_size = CROPSHAPE
  stride = int(CROPSHAPE/3)
  df_test = pd.read_csv('test.csv', delim_whitespace=True, header=0)
  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  sess = tf.InteractiveSession()   
  ###### input data
  gr_data = tf.placeholder(
    tf.float32, [None, data_shape, data_shape, 3], name='gr_data')
  with TowerContext('', is_training=False):
    #logits = gr_models.create_resnet_model(gr_data, is_training=False, isMSC=True, isASPP=True)
    logits = gr_models.create_resnet_model(gr_data, is_training=False)

  logits = tf.image.resize_bilinear(logits, [crop_size, crop_size])
  if CRF:
    image_mean = tf.constant(mean, dtype=tf.float32)
    image_std = tf.constant(std, dtype=tf.float32)
    image_origin = tf.cast((gr_data * image_std + image_mean)*255, tf.uint8)
    logits = tf.nn.softmax(logits) 
    logits = tf.py_func(dense_crf_batch, [logits, image_origin], tf.float32) 
  #softmax ---> resize or resize ---> softmax
  logits = tf.nn.softmax(logits)

  # Load checkpoint
  # assert os.path.isfile(FLAGS.start_checkpoint+'.meta'), FLAGS.start_checkpoint
  gr_models.load_variables_from_checkpoint(sess, 'train_log/imagenet-resnet-d101-onlyval/model-1211488')

  starttime = datetime.datetime.now()
  print('Start eval @ ', starttime.strftime("%Y-%m-%d %H:%M:%S"))

  for start in xrange(0, len(df_test), batch_size):
    x_batch = []
    end = min(start + batch_size, len(df_test))
    df_test_batch = df_test[start:end]
    img_size = []
    real_batch = end - start
    for id in df_test_batch['data']:
      print ('/home/grwang/seg/voc-data/' + id)
      img_ori = cv2.imread('/home/grwang/seg/' + id)
      h_ori, w_ori, _ = img_ori.shape

      if h_ori*w_ori < 460 * 500:
        scs = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]  
      else:
        scs = [0.5, 0.75, 1.0, 1.25, 1.5]
        print(max(h_ori, w_ori))
      

      maps = []
      for sc in scs:
        img = cv2.resize(img_ori, (int(float(w_ori)*sc), int(float(h_ori)*sc)), interpolation=cv2.INTER_CUBIC)
        score_map = data_crop_test_output(sess, gr_data, logits, img, mean, std, crop_size, stride)
        score_map = cv2.resize(score_map, (w_ori, h_ori), interpolation=cv2.INTER_CUBIC)
        maps.append(score_map)
      score_map = np.mean(np.stack(maps), axis=0)

      maps2 = []
      for sc in scs:
        img2 = cv2.resize(img_ori, (int(float(w_ori)*sc), int(float(h_ori)*sc)), interpolation=cv2.INTER_CUBIC)
        img2 = cv2.flip(img2, 1)
        score_map2 = data_crop_test_output(sess, gr_data, logits, img2, mean, std, crop_size, stride)
        score_map2 = cv2.resize(score_map2, (w_ori, h_ori), interpolation=cv2.INTER_CUBIC)
        maps2.append(score_map2)
      score_map2 = np.mean(np.stack(maps2), axis=0)
      score_map2 = cv2.flip(score_map2, 1)
      score_map = (score_map + score_map2)/2

      pred_label = np.argmax(score_map, 2)      
      pred_label = np.asarray(pred_label, dtype='uint8')

      print(np.max(pred_label))
      name = id.split('/', 10)
      cv2.imwrite('prediction-val-flip/' + name[1][0:11] + '.png', pred_label) 
  
  endtime = datetime.datetime.now()
  print('Total time: ', endtime - starttime)


if __name__ == '__main__':
  # if FLAGS.gpu:
  #   os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
  tf.app.run(main=main)
