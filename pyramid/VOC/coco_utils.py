#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: imagenet_utils.py
# Author: Guangrun Wang (wanggrun@mail2.sysu.edu.cn)
# Author: Rongcong Chen (chenrc@mail2.sysu.edu.cn)
# Code is developed based on Yuxin Wu's ResNet implementation: https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet

import cv2
import numpy as np
import multiprocessing
import tensorflow as tf
from abc import abstractmethod

from tensorpack import imgaug, dataset, ModelDesc, InputDesc
from tensorpack.dataflow import (
    MapData, AugmentImageComponent, PrefetchDataZMQ,
    BatchData, MultiThreadMapData)
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor
from tensorpack.utils.stats import RatioCounter
from tensorpack.models import regularize_cost
from tensorpack.tfutils.summary import add_moving_summary

from keras.losses import binary_crossentropy
import keras.backend as K

CROPSHAPE = 448

n_class = 21

def dice_loss(label, logits):
    label = tf.cast(label, tf.float32)
    logits = tf.nn.softmax(logits)
    smooth = 1.
    cc = logits.get_shape().as_list()[1]
    b_ = tf.shape(label)[0]
    grzeros = tf.zeros([b_,])
    grones = tf.ones([b_,])

    grloss = 0
    for c in range(1, cc):

        targetlabel = tf.where(tf.equal(label, c), grones, grzeros)
        targetlogits = logits[:,c]
        intersection = tf.reduce_sum(targetlogits * targetlabel, axis=0)
        grloss = grloss + (2. * intersection + smooth) / (tf.reduce_sum(targetlabel) + tf.reduce_sum(targetlogits) + smooth)
    return grloss

class GoogleNetResize(imgaug.ImageAugmentor):
    """
    crop 8%~100% of the original image
    See `Going Deeper with Convolutions` by Google.
    """
    def __init__(self, crop_area_fraction=0.08,
                 aspect_ratio_low=0.75, aspect_ratio_high=1.333,
                 target_shape=224):
        self._init(locals())

    def _augment(self, img, _):
        h, w = img.shape[:2]
        area = h * w
        for _ in range(10):
            targetArea = self.rng.uniform(self.crop_area_fraction, 1.0) * area
            aspectR = self.rng.uniform(self.aspect_ratio_low, self.aspect_ratio_high)
            ww = int(np.sqrt(targetArea * aspectR) + 0.5)
            hh = int(np.sqrt(targetArea / aspectR) + 0.5)
            if self.rng.uniform() < 0.5:
                ww, hh = hh, ww
            if hh <= h and ww <= w:
                x1 = 0 if w == ww else self.rng.randint(0, w - ww)
                y1 = 0 if h == hh else self.rng.randint(0, h - hh)
                out = img[y1:y1 + hh, x1:x1 + ww]
                out = cv2.resize(out, (self.target_shape, self.target_shape), interpolation=cv2.INTER_CUBIC)
                return out
        out = imgaug.ResizeShortestEdge(self.target_shape, interp=cv2.INTER_CUBIC).augment(img)
        out = imgaug.CenterCrop(self.target_shape).augment(out)
        return out


def fbresnet_augmentor(isTrain):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    if isTrain:
        augmentors = [
            GoogleNetResize(),
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224)),
        ]
    return augmentors

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(0, 0,0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(0, 0,0,))
    return image, mask

def randomScale(img, label):
    scale = np.random.uniform(0.9, 1.1)
    h_old = img.shape[0]
    w_old = img.shape[1]    
    h_new = int(float(h_old) * scale)
    w_new = int(float(w_old) * scale)
    img = cv2.resize(img, (w_new, h_new))
    label = cv2.resize(label, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
    return img, label

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    return image, mask


def randomCropPad(image, label, crop_h, crop_w, ignore_label=255, isTrain=True, isSqueeze=True):
    label = np.asarray(label, dtype='float32')
    label = np.expand_dims(label, -1)
    label = label - float(ignore_label)
    combined = np.concatenate((image, label),axis=2)
    image_h = image.shape[0]
    image_w = image.shape[1]    
    pad_h = max(0, crop_h - image_h)
    pad_w = max(0, crop_w - image_w)
    combined_pad = np.lib.pad(combined, ((0,pad_h), (0,pad_w), (0,0)), 'constant', 
        constant_values=((128,128), (128,128), (128, 128)))
    comb_h = combined_pad.shape[0]
    comb_w = combined_pad.shape[1]  
    if isTrain:
        crop_scale = np.random.uniform(0.0, 1.0)
    else:
        crop_scale = 0.5
    crop_shift_h = int(crop_scale * float((comb_h - crop_h))/2)
    crop_shift_w = int(crop_scale * float((comb_w - crop_w))/2)
    combined_crop = combined_pad[crop_shift_h:crop_shift_h+crop_h, crop_shift_w:crop_shift_w+crop_w, :]
    img_crop = combined_crop[:, :, :3]
    label_crop = combined_crop[:, :, 3:]
    label_crop = label_crop + ignore_label
    label_crop = np.asarray(label_crop, dtype='uint8')
    
    # Set static shape so that tensorflow knows shape at compile time. 
    # img_crop.set_shape((crop_h, crop_w, 3))
    # label_crop.set_shape((crop_h,crop_w, 1))
    if isSqueeze:
    	label_crop = np.squeeze(label_crop, -1)
    return img_crop, label_crop


def get_voc_dataflow(
        datadir, name, batch_size,
        augmentors, isDownSample=True, isSqueeze=True, isExpandDim=False):
    """
    See explanations in the tutorial:
    http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html
    """
    gr_shape = CROPSHAPE
    assert name in ['train', 'val', 'test']
    assert datadir is not None
    assert isinstance(augmentors, list)
    isTrain = name == 'train'
    cpu = min(30, multiprocessing.cpu_count())
    if isTrain:
        ds = dataset.VOC12(datadir, name, shuffle=True)
        # ds = AugmentImageComponent(ds, augmentors, copy=False)
        aug = imgaug.AugmentorList(augmentors)
        def preprocess(imandlabel):

            im, label = imandlabel
            assert im is not None, 'aaaaaa'
            assert label is not None, 'bbbbbbb'

            im = im.astype('float32')
            label = label.astype('uint8')

            im = randomHueSaturationValue(im, hue_shift_limit=(-50, 50), sat_shift_limit=(-50, 50), val_shift_limit=(-50, 50))

            im, label = randomScale(im, label)
            im, label = randomHorizontalFlip(im, label)
            im, label = randomCropPad(im, label, gr_shape, gr_shape, ignore_label=255, isTrain=isTrain, isSqueeze=isSqueeze)                        
            ret = [im]
            if isDownSample:
            	label = cv2.resize(label, (gr_shape/8, gr_shape/8), interpolation=cv2.INTER_NEAREST)
            label = np.asarray(label, dtype='uint8')
            ret.append(label)
            return ret
        ds = MapData(ds, preprocess)
        ds = PrefetchDataZMQ(ds, cpu)
        ds = BatchData(ds, batch_size, remainder=False)
    else:
        ds = dataset.VOC12Files(datadir, name, shuffle=False)
        aug = imgaug.AugmentorList(augmentors)

        def mapf(dp):
            dataname, labelname = dp
            im = cv2.imread(dataname, cv2.IMREAD_COLOR)            
            label = cv2.imread(labelname, cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, (gr_shape, gr_shape))
            label = cv2.resize(label, (gr_shape, gr_shape), interpolation=cv2.INTER_NEAREST)
            if isExpandDim:
            	label = np.expand_dims(label, -1)


            if isDownSample:         
            	label = cv2.resize(label, (gr_shape/8, gr_shape/8), interpolation=cv2.INTER_NEAREST)
            label = np.asarray(label, dtype='uint8')

            return im, label
        #ds = MultiThreadMapData(ds, cpu, mapf, buffer_size=1000, strict=True)
        ds = MultiThreadMapData(ds, cpu, mapf, buffer_size=100, strict=True)
        ds = BatchData(ds, batch_size, remainder=True)
        ds = PrefetchDataZMQ(ds, 1)
    return ds


def get_imagenet_dataflow(
        datadir, name, batch_size,
        augmentors):
    """
    See explanations in the tutorial:
    http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html
    """
    assert name in ['train', 'val', 'test']
    assert datadir is not None
    assert isinstance(augmentors, list)
    isTrain = name == 'train'
    cpu = min(30, multiprocessing.cpu_count())
    if isTrain:
        ds = dataset.ILSVRC12(datadir, name, shuffle=True)
        ds = AugmentImageComponent(ds, augmentors, copy=False)
        ds = PrefetchDataZMQ(ds, cpu)
        ds = BatchData(ds, batch_size, remainder=False)
    else:
        ds = dataset.ILSVRC12Files(datadir, name, shuffle=False)
        aug = imgaug.AugmentorList(augmentors)

        def mapf(dp):
            fname, cls = dp
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = aug.augment(im)
            return im, cls
        ds = MultiThreadMapData(ds, cpu, mapf, buffer_size=2000, strict=True)
        ds = BatchData(ds, batch_size, remainder=True)
        ds = PrefetchDataZMQ(ds, 1)
    return ds


def eval_on_ILSVRC12(model, sessinit, dataflow):
    pred_config = PredictConfig(
        model=model,
        session_init=sessinit,
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'wrong-top5']
    )
    pred = SimpleDatasetPredictor(pred_config, dataflow)
    acc1, acc5 = RatioCounter(), RatioCounter()
    for top1, top5 in pred.get_result():
        batch_size = top1.shape[0]
        acc1.feed(top1.sum(), batch_size)
        acc5.feed(top5.sum(), batch_size)
    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))


class ImageNetModel(ModelDesc):
    weight_decay = 1e-4
    image_shape = CROPSHAPE

    """
    uint8 instead of float32 is used as input type to reduce copy overhead.
    It might hurt the performance a liiiitle bit.
    The pretrained models were trained with float32.
    """
    image_dtype = tf.uint8

    def __init__(self, data_format='NCHW', hardmine=False):
        self.data_format = data_format
        self.hardmine = hardmine

    def _get_inputs(self):
        return [InputDesc(self.image_dtype, [None, self.image_shape, self.image_shape, 3], 'input'),
                InputDesc(tf.int32, [None, self.image_shape /8, self.image_shape/8], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = self.image_preprocess(image, bgr=True)
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])

        logits = self.get_logits(image)

        print(type(logits), type(label), logits.get_shape().as_list(), label.get_shape().as_list())

        label = tf.reshape(label, [-1])
        logits = tf.reshape(logits, [-1, logits.get_shape().as_list()[3]])

        #######
        if self.hardmine:
            print('--------------------hardmine-----------------------')
            def process_hardmine_class(logits, label, class_number):
                indices = tf.squeeze(tf.where(tf.equal(label, 2)), 1)
                label = tf.cast(tf.gather(label, indices), tf.int32)
                logits = tf.gather(logits, indices)
                return logits, label

            logits_bike, label_bike = process_hardmine_class(logits, label, 2)
            logits_boat, label_boat = process_hardmine_class(logits, label, 4)

            logits_hard = tf.concat(values=[logits_bike, logits_boat] * 1, axis = 0)
            label_hard = tf.concat(values=[label_bike, label_boat] * 1, axis = 0)
            indices = tf.squeeze(tf.where(tf.less_equal(label, n_class-1)), 1)
            label = tf.cast(tf.gather(label, indices), tf.int32)
            logits = tf.gather(logits, indices)
            label = tf.concat(values = [label, label_hard], axis = 0)
            logits = tf.concat(values = [logits, logits_hard], axis = 0)
            print('--------------------hardmine-----------------------')
        else:
            indices = tf.squeeze(tf.where(tf.less_equal(label, n_class-1)), 1)
            label = tf.cast(tf.gather(label, indices), tf.int32)
            logits = tf.gather(logits, indices)

        loss = ImageNetModel.compute_loss_and_error(logits, label)

        if self.weight_decay > 0:
            wd_loss = regularize_cost('.*/W', tf.contrib.layers.l2_regularizer(self.weight_decay),
                                      name='l2_regularize_loss')
            add_moving_summary(loss, wd_loss)
            self.cost = tf.add_n([loss, wd_loss], name='cost')
        else:
            self.cost = tf.identity(loss, name='cost')
            add_moving_summary(self.cost)

    @abstractmethod
    def get_logits(self, image):
        """
        Args:
            image: 4D tensor of 224x224 in ``self.data_format``

        Returns:
            Nx1000 logits
        """

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

    def image_preprocess(self, image, bgr=True):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            image = image * (1.0 / 255)

            mean = [0.485, 0.456, 0.406]    # rgb
            std = [0.229, 0.224, 0.225]
            if bgr:
                mean = mean[::-1]
                std = std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32)
            image_std = tf.constant(std, dtype=tf.float32)
            image = (image - image_mean) / image_std
            return image

    @staticmethod
    def compute_loss_and_error(logits, label):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        loss = tf.reduce_mean(loss, name='xentropy-loss')
        # loss = loss + (1 - dice_loss(label, logits))

        def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
            with tf.name_scope('prediction_incorrect'):
                x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
            return tf.cast(x, tf.float32, name=name)

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))
        return loss


class MSC_Model(ModelDesc):
    weight_decay = 1e-4
    image_shape = CROPSHAPE

    """
    uint8 instead of float32 is used as input type to reduce copy overhead.
    It might hurt the performance a liiiitle bit.
    The pretrained models were trained with float32.
    """
    image_dtype = tf.uint8

    def __init__(self, data_format='NCHW', hardmine=False):
        self.data_format = data_format
        self.hardmine = hardmine

    def _get_inputs(self):
        return [InputDesc(self.image_dtype, [None, self.image_shape, self.image_shape, 3], 'input'),
                InputDesc(tf.int32, [None, self.image_shape, self.image_shape, 1], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = self.image_preprocess(image, bgr=True)
        image075 = tf.image.resize_images(image, [int(image.get_shape().as_list()[1]*0.75), int(image.get_shape().as_list()[2]*0.75)])
        image05 = tf.image.resize_images(image, [int(image.get_shape().as_list()[1]*0.5), int(image.get_shape().as_list()[2]*0.5)])
        if self.data_format == 'NCHW':
            image = tf.transpose(image, [0, 3, 1, 2])

        logits100 = self.get_logits(image)
        with tf.variable_scope('', reuse=True):
            logits075 = self.get_logits(image075)
        with tf.variable_scope('', reuse=True):
            logits05 = self.get_logits(image05)

        logits = tf.reduce_max(tf.stack([logits100, tf.image.resize_images(logits075, tf.shape(logits100)[1:3, ]), tf.image.resize_images(logits05, tf.shape(logits100)[1:3, ])]), axis=0)
        #logits = self.get_logits(image)

        def process_label(logits, label): 
            label = tf.image.resize_nearest_neighbor(label, logits.get_shape()[1:3])
            print(type(logits), type(label), logits.get_shape().as_list(), label.get_shape().as_list())
            label = tf.squeeze(label, -1)
            label = tf.reshape(label, [-1])
            return label
   
        label075 = process_label(logits075, label)
        label05 = process_label(logits05, label)
        label = process_label(logits, label)

        logits = tf.reshape(logits, [-1, logits.get_shape().as_list()[3]])
        logits100 = tf.reshape(logits100, [-1, logits100.get_shape().as_list()[3]])
        logits075 = tf.reshape(logits075, [-1, logits075.get_shape().as_list()[3]])
        logits05 = tf.reshape(logits05, [-1, logits05.get_shape().as_list()[3]])


        #######
        self.hardmine = False
        if self.hardmine:
            print('--------------------hardmine-----------------------')
            def process_hardmine_class(logits, label, class_number):
                indices = tf.squeeze(tf.where(tf.equal(label, 2)), 1)
                label = tf.cast(tf.gather(label, indices), tf.int32)
                logits = tf.gather(logits, indices)
                return logits, label

            # logits_bike, label_bike = process_hardmine_class(logits, label, 2)
            # logits100_bike, _ = process_hardmine_class(logits100, label, 2)
            # logits075_bike, label075_bike = process_hardmine_class(logits075, label075, 2)
            # logits05_bike, label05_bike = process_hardmine_class(logits05, label05, 2)

            # logits_boat, label_boat = process_hardmine_class(logits, label, 4)
            # logits100_boat, _ = process_hardmine_class(logits100, label, 4)
            # logits075_boat, label075_boat = process_hardmine_class(logits075, label075, 4)
            # logits05_boat, label05_boat = process_hardmine_class(logits05, label05, 4)
            '''
            indices_bike = tf.squeeze(tf.where(tf.equal(raw_label, 2)), 1)
            label_bike = tf.cast(tf.gather(raw_label, indices_bike), tf.int32)
            logits_bike = tf.gather(raw_logits, indices_bike)

            indices_boat = tf.squeeze(tf.where(tf.equal(raw_label, 4)), 1)
            label_boat = tf.cast(tf.gather(raw_label, indices_boat), tf.int32)
            logits_boat = tf.gather(raw_logits, indices_boat)


            indices_chair = tf.squeeze(tf.where(tf.equal(raw_label, 9)), 1)
            label_chair = tf.cast(tf.gather(raw_label, indices_chair), tf.int32)
            logits_chair = tf.gather(raw_logits, indices_chair)


            indices_table = tf.squeeze(tf.where(tf.equal(raw_label, 11)), 1)
            label_table = tf.cast(tf.gather(raw_label, indices_table), tf.int32)
            logits_table = tf.gather(raw_logits, indices_table)


            indices_plant = tf.squeeze(tf.where(tf.equal(raw_label, 16)), 1)
            label_plant = tf.cast(tf.gather(raw_label, indices_plant), tf.int32)
            logits_plant = tf.gather(raw_logits, indices_plant)


            indices_sofa = tf.squeeze(tf.where(tf.equal(raw_label, 18)), 1)
            label_sofa = tf.cast(tf.gather(raw_label, indices_sofa), tf.int32)
            logits_sofa = tf.gather(raw_logits, indices_sofa)

            logits_hard = tf.concat(values=[logits_bike, logits_bike, logits_boat, logits_chair, logits_chair, 
                logits_table, logits_table, logits_plant, logits_sofa, logits_sofa] * 1, axis = 0)
            label_hard = tf.concat(values=[label_bike, label_bike, label_boat, label_chair, label_chair,
                label_table, label_table, label_plant, label_sofa, label_sofa] * 1, axis = 0)
            '''
            logits_hard = tf.concat(values=[logits_bike, logits_boat] * 1, axis = 0)
            logits100_hard = tf.concat(values=[logits100_bike, logits100_boat] * 1, axis = 0)
            logits075_hard = tf.concat(values=[logits075_bike, logits075_boat] * 1, axis = 0)
            logits05_hard = tf.concat(values=[logits05_bike, logits05_boat] * 1, axis = 0)

            label_hard = tf.concat(values=[label_bike, label_boat] * 1, axis = 0)
            label075_hard = tf.concat(values=[label075_bike, label075_boat] * 1, axis = 0)
            label05_hard = tf.concat(values=[label05_bike, label05_boat] * 1, axis = 0)


            indices = tf.squeeze(tf.where(tf.less_equal(label, n_class-1)), 1)
            indices075 = tf.squeeze(tf.where(tf.less_equal(label075, n_class-1)), 1)
            indices05 = tf.squeeze(tf.where(tf.less_equal(label05, n_class-1)), 1)

            label = tf.cast(tf.gather(label, indices), tf.int32)
            label075 = tf.cast(tf.gather(label075, indices075), tf.int32)
            label05 = tf.cast(tf.gather(label05, indices05), tf.int32)

            logits = tf.gather(logits, indices)
            logits100 = tf.gather(logits100, indices)
            logits075 = tf.gather(logits075, indices075)
            logits05 = tf.gather(logits05, indices05)

            label = tf.concat(values = [label, label_hard], axis = 0)
            label075 = tf.concat(values = [label075, label075_hard], axis = 0)
            label05 = tf.concat(values = [label05, label05_hard], axis = 0)

            logits = tf.concat(values = [logits, logits_hard], axis = 0)
            logits100 = tf.concat(values = [logits100, logits100_hard], axis = 0)
            logits075 = tf.concat(values = [logits075, logits075_hard], axis = 0)
            logits05 = tf.concat(values = [logits05, logits05_hard], axis = 0)
            print('--------------------hardmine-----------------------')
        else:
            indices = tf.squeeze(tf.where(tf.less_equal(label, n_class-1)), 1)
            indices075 = tf.squeeze(tf.where(tf.less_equal(label075, n_class-1)), 1)
            indices05 = tf.squeeze(tf.where(tf.less_equal(label05, n_class-1)), 1)

            label = tf.cast(tf.gather(label, indices), tf.int32)
            label075 = tf.cast(tf.gather(label075, indices075), tf.int32)
            label05 = tf.cast(tf.gather(label05, indices05), tf.int32)

            logits = tf.gather(logits, indices)
            logits100 = tf.gather(logits100, indices)
            logits075 = tf.gather(logits075, indices075)
            logits05 = tf.gather(logits05, indices05)

        loss = MSC_Model.compute_loss_and_error(logits, label, prefix='loss-')
        loss100 = MSC_Model.compute_loss_and_error(logits100, label, prefix='loss100-')
        loss075 = MSC_Model.compute_loss_and_error(logits075, label075, prefix='loss075-')
        loss05 = MSC_Model.compute_loss_and_error(logits05, label05, prefix='loss05-')

        if self.weight_decay > 0:
            wd_loss = regularize_cost('.*/W', tf.contrib.layers.l2_regularizer(self.weight_decay),
                                      name='l2_regularize_loss')
            add_moving_summary(loss,loss100,loss075,loss05, wd_loss)
            self.cost = tf.add_n([loss, loss100, loss075, loss05, wd_loss], name='cost')
        else:
            add_moving_summary(loss,loss100,loss075,loss05)
            self.cost = tf.add_n([loss, loss100, loss075, loss05], name='cost')

    @abstractmethod
    def get_logits(self, image):
        """
        Args:
            image: 4D tensor of 224x224 in ``self.data_format``

        Returns:
            Nx1000 logits
        """

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

    def image_preprocess(self, image, bgr=True):
        with tf.name_scope('image_preprocess'):
            if image.dtype.base_dtype != tf.float32:
                image = tf.cast(image, tf.float32)
            image = image * (1.0 / 255)

            mean = [0.485, 0.456, 0.406]    # rgb
            std = [0.229, 0.224, 0.225]
            if bgr:
                mean = mean[::-1]
                std = std[::-1]
            image_mean = tf.constant(mean, dtype=tf.float32)
            image_std = tf.constant(std, dtype=tf.float32)
            image = (image - image_mean) / image_std
            return image

    @staticmethod
    def compute_loss_and_error(logits, label, prefix=''):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        loss = tf.reduce_mean(loss, name=prefix+'xentropy-loss')
        # loss = loss + (1 - dice_loss(label, logits))

        def prediction_incorrect(logits, label, topk=1, name='incorrect_vector'):
            with tf.name_scope('prediction_incorrect'):
                x = tf.logical_not(tf.nn.in_top_k(logits, label, topk))
            return tf.cast(x, tf.float32, name=name)

        wrong = prediction_incorrect(logits, label, 1, name=prefix+'wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name=prefix+'train-error-top1'))

        wrong = prediction_incorrect(logits, label, 5, name=prefix+'wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name=prefix+'train-error-top5'))
        return loss
