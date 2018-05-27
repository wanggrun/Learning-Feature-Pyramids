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

CROPSHAPE = 512

FLAGS = None
os.environ['CUDA_VISIBLE_DEVICES'] = "6"

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
            image_crop_batch.append(image_pad[x:x+crop_size, y:y+crop_size, :])
    logits = session.run(
        [logits],
        feed_dict={
            gr_data: image_crop_batch,
        })
    logits = np.reshape(logits[0], [len(image_crop_batch), int(CROPSHAPE/8), int(CROPSHAPE/8), 21])
    num_class = 21
    score_map = np.zeros([image_pad.shape[0], image_pad.shape[1], num_class], dtype = 'float32')
    count = np.zeros([image_pad.shape[0], image_pad.shape[1], num_class], dtype = 'float32')
    crop_index = 0
    for x in x_start:
        for y in y_start:
    	    crop_logits = cv2.resize(logits[crop_index], (crop_size, crop_size))
            score_map[x:x+crop_logits.shape[0], y:y+crop_logits.shape[1], :] += crop_logits
            count[x:x+crop_logits.shape[0], y:y+crop_logits.shape[1], :] += 1 
            crop_index += 1
     
    score_map = score_map[:image_h,:image_w, :] / count[:image_h,:image_w, :]
    score_vector = np.reshape(score_map, [score_map.shape[0]*score_map.shape[1], score_map.shape[2]])
    predicted_pixel_label = np.argmax(score_vector, 1)
    predicted_label = np.reshape(predicted_pixel_label, [score_map.shape[0], score_map.shape[1]])
    predicted_label = np.asarray(predicted_label, dtype='uint8')
    return predicted_label

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
  df_test = pd.read_csv('val.csv', delim_whitespace=True, header=0)
  # print (df_test['data'])
  # ids_test = df_test['img'].map(lambda s: s.split('.')[0])
  # names = []
  # for id in ids_test:
  #   names.append('{}.jpg'.format(id))

  # We want to see all the logging messages for this tutorial.
  tf.logging.set_verbosity(tf.logging.INFO)

  # Start a new TensorFlow session.
  sess = tf.InteractiveSession()   
  ###### input data
  data_shape = CROPSHAPE
  gr_data = tf.placeholder(
    tf.float32, [None, data_shape, data_shape, 3], name='gr_data')
  with TowerContext('', is_training=False):
    logits = gr_models.create_resnet_model(gr_data, is_training=False, isMSC=True, isASPP=True)

  input_shape =logits.get_shape().as_list()
  b_ = tf.shape(logits)[0] * input_shape[1] * input_shape[2]
  logits = tf.reshape(logits, [b_, input_shape[3]])
  logits = tf.nn.softmax(logits)
  #logits = tf.reshape(logits, [None, 56, 56, 21])
  #predicted_indices = tf.argmax(logits, 1)
     

  # Define loss and optimizer
  gr_label = tf.placeholder(
      tf.int64, [None], name='gr_label')

  # Optionally we can add runtime checks to spot when NaNs or other symptoms of
  # numerical errors start occurring during training.
  control_dependencies = []
  # if FLAGS.check_nans:
  checks = tf.add_check_numerics_ops()
  control_dependencies = [checks]
  

  # Load checkpoint
  # assert os.path.isfile(FLAGS.start_checkpoint+'.meta'), FLAGS.start_checkpoint
  gr_models.load_variables_from_checkpoint(sess, 'train_log/imagenet-resnet-d101-voc-JT-bk/model-1131572')

  # start_step = global_step.eval(session=sess)
  # tf.logging.info('Testing from step: %d ', start_step)

  fname = []
  output = []
  # mean = [0.406, 0.485, 0.456]    
  # std = [0.225, 0.229, 0.224]

  mean = [0.406, 0.456, 0.485]
  std =  [0.225, 0.224, 0.229]

  mean = np.asarray(mean, dtype='float32')
  std = np.asarray(std, dtype='float32')
  crop_size = CROPSHAPE
  stride = int(CROPSHAPE/10)
  # print('Start test @ ', starttime.strftime("%Y-%m-%d %H:%M:%S"))

  for start in xrange(0, len(df_test), batch_size):
    x_batch = []
    end = min(start + batch_size, len(df_test))
    df_test_batch = df_test[start:end]
    img_size = []
    real_batch = end - start
    for id in df_test_batch['data']:
      print (id)
      img = cv2.imread('/media/SSD/wyang/dataset/mscoco/' + id, cv2.IMREAD_COLOR)      
      pred_label = data_crop_test_output(sess, gr_data, logits, img, mean, std, crop_size, stride)
      # print(np.max(pred_label))
      name = id.split('/', 10)
      cv2.imwrite('prediction/' + name[1][0:11] + '.png', pred_label) 
  
  # # endtime = datetime.datetime.now()
  # # print('Total time: ', endtime - starttime)



if __name__ == '__main__':
  # if FLAGS.gpu:
  #   os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
  tf.app.run(main=main)
