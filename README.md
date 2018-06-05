# Train ImageNet and PASCAL VOC2012 \\ via Learning Feature Pyramids

The codes are  provided by [Guangrun Wang](https://wanggrun.github.io/)

Sun Yat-sen University (SYSU)

### Table of Contents
0. [Introduction](#introduction)
0. [Citation](#citation)

### Introduction

This repository contains the models described in the paper "Learning Feature Pyramids for Human Pose Estimation" (https://arxiv.org/abs/1708.01101). These models are those used in [ILSVRC](http://image-net.org/challenges/LSVRC/2015/)


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

