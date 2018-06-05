# Training ImageNet and PASCAL VOC2012 via Learning Feature Pyramids

The code is provided by [Guangrun Wang](https://wanggrun.github.io/).

Sun Yat-sen University (SYSU)

### Table of Contents
0. [Introduction](#introduction)
0. [ImageNet](#imagenet)
0. [PASCAL VOC2012](#voc)
0. [Citation](#citation)

### Introduction

This repository contains the training & testing code on [ImageNet](http://image-net.org/challenges/LSVRC/2015/) and [PASCAL VOC2012](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6) via learning feature pyramids (LFP). LFP is originally used for human pose machine, described in the paper "Learning Feature Pyramids for Human Pose Estimation" (https://arxiv.org/abs/1708.01101). We extend it to the semantic image segmentation.

### ImageNet

First:
+ cd pyramid/ImageNet/ 

Training script:
+ python resnet-msc-voc-aspp.py   --gpu 0,1,2,3,4,5,6,7  --load ../train_log_trainval/imagenet-resnet-d101-trainval/model-1187596  --data_format NHWC  -d 101  --mode resnet --log_dir onlyval  --data  /media/SSD/wyang/datase

Testing script:
+ python resnet-msc-voc-aspp.py   --gpu 0,1,2,3,4,5,6,7  --load ../train_log_trainval/imagenet-resnet-d101-trainval/model-1187596  --data_format NHWC  -d 101  --mode resnet --log_dir onlyval  --data  /media/SSD/wyang/datase

Trained Models:


### PASCAL VOC2012

First:
+ cd pyramid/VOC/

Training script:
+ python resnet-msc-voc-aspp.py   --gpu 0,1,2,3,4,5,6,7  --load ../train_log_trainval/imagenet-resnet-d101-trainval/model-1187596  --data_format NHWC  -d 101  --mode resnet --log_dir onlyval  --data  /media/SSD/wyang/datase

Testing script:
+ python gr_test_pad_crf_msc_flip.py 

Trained Models

### Citation

If you use these models in your research, please cite:

	@inproceedings{yang2017learning,
            title={Learning feature pyramids for human pose estimation},
            author={Yang, Wei and Li, Shuang and Ouyang, Wanli and Li, Hongsheng and Wang, Xiaogang},
            booktitle={The IEEE International Conference on Computer Vision (ICCV)},
            volume={2},
            year={2017}
        }

### Dependencies
+ Python 2.7 or 3
+ TensorFlow >= 1.3.0
+ [Tensorpack](https://github.com/ppwwyyxx/tensorpack)
   The code depends on Yuxin Wu's Tensorpack. For convenience, we provide a stable version 'tensorpack-installed' in this repository. 
   ```
   # install tensorpack locally:
   cd tensorpack-installed
   python setup.py install --user
   ```

