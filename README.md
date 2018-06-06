# Training ImageNet and PASCAL VOC2012 via Learning Feature Pyramids

The code is provided by [Guangrun Wang](https://wanggrun.github.io/) ([Rongcong Chen](http://www.sysu-hcp.net/people/) also provides contributions).

Sun Yat-sen University (SYSU)

### Table of Contents
0. [Introduction](#introduction)
0. [ImageNet](#imagenet)
0. [PASCAL VOC2012](#voc)
0. [Citation](#citation)

### Introduction

This repository contains the training & testing code on [ImageNet](http://image-net.org/challenges/LSVRC/2015/) and [PASCAL VOC2012](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6) via learning feature pyramids (LFP). LFP is originally used for human pose machine, described in the paper "Learning Feature Pyramids for Human Pose Estimation" (https://arxiv.org/abs/1708.01101). We extend it to the semantic image segmentation.

### ImageNet

+ First:
```
cd pyramid/ImageNet/ 
```

+ Training script:
```
python imagenet-resnet.py   --gpu 0,1,2,3,4,5,6,7   --data_format NHWC  -d 101  --mode resnet --data  [ROOT-OF-IMAGENET-DATASET]
```

+ Testing script:
```
python imagenet-resnet.py   --gpu 0,1,2,3,4,5,6,7  --load [ROOT-TO-LOAD-MODEL]  --data_format NHWC  -d 101  --mode resnet --data  [ROOT-OF-IMAGENET-DATASET] --eval
```

+ Trained Models:

   [Baidu Pan](https://wanggrun.github.io/)

   [Google Drive](https://wanggrun.github.io/)

### PASCAL VOC2012

+ First:
```
cd pyramid/VOC/
```

+ Training script:
```
# Use the ImageNet classification model as pretrained model.
# Because ImageNet has 1,000 categories while voc only has 21 categories, 
# we must first fix all the parameters except the last layer including 21 channels. We only train the last layer for adaption
# by adding: "with freeze_variables(stop_gradient=True, skip_collection=True): " in Line 206 of resnet_model_voc_aspp.py
# Then we finetune all the parameters.
# For evaluation on voc val set, the model is first trained on COCO, then on train_aug of voc. 
# For evaluation on voc leaderboard (test set), the above model is further trained on voc val.
# it achieves 81.0% on voc leaderboard.
# a training script example is as follows.
python resnet-msc-voc-aspp.py   --gpu 0,1,2,3,4,5,6,7  --load [ROOT-TO-LOAD-MODEL]  --data_format NHWC  -d 101  --mode resnet --log_dir [ROOT-TO-SAVE-MODEL]  --data [ROOT-OF-TRAINING-DATA]
```

+ Testing script:
```
python gr_test_pad_crf_msc_flip.py 
```

+ Trained Models:

   Model trained for evaluation on voc val set:

   [Baidu Pan](https://wanggrun.github.io/)

   [Google Drive](https://wanggrun.github.io/)

   Model trained for evaluation on voc leaderboard (test set)

   [Baidu Pan](https://wanggrun.github.io/)

   [Google Drive](https://wanggrun.github.io/)

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

