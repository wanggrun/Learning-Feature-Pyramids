#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: voc.py
# Author: Guangrun Wang <wanggrun@mail.sysu.edu.cn>
import os
import tarfile
import numpy as np
import tqdm

from ...utils import logger
from ...utils.loadcaffe import get_caffe_pb
from ...utils.fs import mkdir_p, download, get_dataset_path
from ...utils.timer import timed_operation
from ..base import RNGDataFlow

__all__ = ['VOC12', 'VOC12Files']



class VOC12Files(RNGDataFlow):
    """
    Same as :class:`VOC12`, but produces filenames of the images instead of nparrays.
    This could be useful when ``cv2.imread`` is a bottleneck and you want to
    decode it in smarter ways (e.g. in parallel).
    """
    def __init__(self, dir, name, meta_dir=None,
                 shuffle=None, dir_structure=None):
        """
        Same as in :class:`VOC12`.
        """
        assert name in ['train', 'test', 'val'], name
        assert os.path.isdir(dir), dir
        self.full_dir = dir
        # self.full_dir = os.path.join(dir, name)
        self.name = name
        assert os.path.isdir(self.full_dir), self.full_dir
        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle

        if name == 'train':
            dir_structure = 'train'


        self.imglist = []
        fname = name + '.txt'
        assert os.path.isfile(fname), fname

        with open(fname) as f:
            for line in f.readlines():
                dataname, labelname = line.strip().split()
                self.imglist.append((dataname.strip(), labelname.strip()))
        assert len(self.imglist), fname

        #### check if the file name is wrong
        for dataname, _ in self.imglist[:10]:
            dataname = os.path.join(self.full_dir, dataname)
            assert os.path.isfile(dataname), dataname


    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            dataname, labelname = self.imglist[k]
            dataname = os.path.join(self.full_dir, dataname)
            labelname = os.path.join(self.full_dir, labelname)
            yield [dataname, labelname]


class VOC12(VOC12Files):
    """
    Produces uint8 VOC12 images of shape [h, w, 3(BGR)], and a label between [0, 999].
    """
    def __init__(self, dir, name, meta_dir=None,
                 shuffle=None, dir_structure=None):        
        super(VOC12, self).__init__(
            dir, name, meta_dir, shuffle, dir_structure)

    def get_data(self):
        for dataname, labelname in super(VOC12, self).get_data():
            # print dataname, labelname
            im = cv2.imread(dataname, cv2.IMREAD_COLOR)
            label = cv2.imread(labelname, cv2.IMREAD_GRAYSCALE)
            # print(label)
            assert im is not None, dataname
            yield [im, label]

    


try:
    import cv2
except ImportError:
    from ...utils.develop import create_dummy_class
    VOC12 = create_dummy_class('VOC12', 'cv2')  # noqa

if __name__ == '__main__':
    # meta = VOCMeta()
    # print(meta.get_synset_words_1000())

    ds = VOC12('/home/grwang/tensorpack_seg/tensorpack_gr/example/ResNet/', 'train', shuffle=False)
    ds.reset_state()

    for k in ds.get_data():
        from IPython import embed
        embed()
        break
