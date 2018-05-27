#!~/anaconda2/bin python
# -*- coding: UTF-8 -*-
# File: resnet-msc-coco.py
# Author: Guangrun Wang (wanggrun@mail2.sysu.edu.cn)

import argparse
import os
import sys
import os


from tensorpack import logger, QueueInput
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.train import (TrainConfig, SyncMultiGPUTrainerParameterServer, launch_train_with_config)
from tensorpack.dataflow import FakeData
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.utils.gpu import get_nr_gpu

from coco_utils import (
    fbresnet_augmentor, get_voc_dataflow, MSC_Model,
    eval_on_ILSVRC12)
from resnet_model_coco import (
    preresnet_group, preresnet_basicblock, preresnet_bottleneck, 
    resnet_group, resnet_group_dilation,resnet_basicblock, resnet_bottleneck, resnet_bottleneck_dilation, se_resnet_bottleneck,
    resnet_backbone)

#Fintune
# TOTAL_BATCH_SIZE = 32
#Joint Train
TOTAL_BATCH_SIZE = 16


class Model(MSC_Model):
    def __init__(self, depth, data_format='NCHW', mode='resnet'):
        super(Model, self).__init__(data_format)

        if mode == 'se':
            assert depth >= 50

        self.mode = mode
        basicblock = preresnet_basicblock if mode == 'preact' else resnet_basicblock
        bottleneck = {
            'resnet': resnet_bottleneck,
            'preact': preresnet_bottleneck,
            'se': se_resnet_bottleneck}[mode]
        self.num_blocks, self.block_func = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }[depth]

    def get_logits(self, image):
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            with argscope([BatchNorm], use_local_stat=False):
                return resnet_backbone(
                    image, self.num_blocks,
                    preresnet_group if self.mode == 'preact' else resnet_group, resnet_group_dilation, 
                    self.block_func, resnet_bottleneck_dilation)#Fintune or Joint Train


def get_data(name, batch):
    isTrain = name == 'train'
    augmentors = fbresnet_augmentor(isTrain)
    return get_voc_dataflow(
        args.data, name, batch, augmentors, isDownSample=False, isSqueeze=False, isExpandDim=True)


def get_config(model, fake=False):
    nr_tower = max(get_nr_gpu(), 1)
    batch = TOTAL_BATCH_SIZE // nr_tower

    if fake:
        logger.info("For benchmark, batch size is fixed to 64 per tower.")
        dataset_train = FakeData(
            [[64, 224, 224, 3], [64]], 1000, random=False, dtype='uint8')
        callbacks = []
    else:
        logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
        dataset_train = get_data('train', batch)
        dataset_val = get_data('val', batch)
        callbacks = [
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(0, 2.5e-4), (20, 1.25e-4), (40, 5e-5), (60, 2.5e-5),(80, 1e-5), (100, 5e-6), (120, 5e-7)]),
            HumanHyperParamSetter('learning_rate'),
        ]
        #Fine-tune 
        # [(0, 2.5e-3), (20, 1.25e-3), (40, 5e-4), (60, 2.5e-4),(80, 1e-4), (100, 5e-5), (120, 5e-6)]
        #Joint Train
        #[(0, 2.5e-4), (20, 1.25e-4), (40, 5e-5), (60, 2.5e-5),(80, 1e-5), (100, 5e-6), (120, 5e-7)]
        #infs = [ClassificationError('wrong-top1', 'val-error-top1'),
        #        ClassificationError('wrong-top5', 'val-error-top5')]
        infs = [ClassificationError('loss-wrong-top1', 'loss-val-error-top1'),
                ClassificationError('loss-wrong-top5', 'loss-val-error-top5')]
        if nr_tower == 1:
            # single-GPU inference with queue prefetch
            callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
        else:
            # multi-GPU inference (with mandatory queue prefetch)
            callbacks.append(DataParallelInferenceRunner(
                dataset_val, infs, list(range(nr_tower))))

    return TrainConfig(
        model=model,
        dataflow=dataset_train,
        callbacks=callbacks,
        steps_per_epoch=1522,
        max_epoch=140,
        nr_tower=nr_tower
    )
    #Joint Train
    #steps_per_epoch=3043,
    #max_epoch=50,


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--fake', help='use fakedata to test or benchmark this model', action='store_true')
    parser.add_argument('--data_format', help='specify NCHW or NHWC',
                        type=str, default='NCHW')
    parser.add_argument('-d', '--depth', help='resnet depth',
                        type=int, default=18, choices=[18, 34, 50, 101, 152])
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--mode', choices=['resnet', 'preact', 'se'],
                        help='variants of resnet to use', default='resnet')
    parser.add_argument('--log_dir', help='save model dir')

    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = Model(args.depth, args.data_format, args.mode)
    if args.eval:
        batch = 16    # something that can run on one gpu
        ds = get_data('val', batch)
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)
    else:
        logger.set_logger_dir(
            os.path.join('train_log', 'imagenet-resnet-d' + str(args.depth)+ "-" + args.log_dir))
        #set your save path

        config = get_config(model, fake=args.fake)
        if args.load:
            config.session_init = get_model_loader(args.load)
        trainer = SyncMultiGPUTrainerParameterServer(max(get_nr_gpu(), 1))
        launch_train_with_config(config, trainer)
