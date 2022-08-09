# The 8-Point Algorithm as an Inductive Bias for Relative Pose Prediction by ViTs (3DV 2022)

**Chris Rockwell**, **Justin Johnson** and **David F. Fouhey**

[Project Website](https://crockwell.github.io/rel_pose/) | [Paper](https://crockwell.github.io/rel_pose/data/paper.pdf) |
[Supplemental](https://crockwell.github.io/rel_pose/data/supp.pdf)

<img src="teaser.png" alt="drawing">

## Overview
We propose three small modifications to a ViT via the Essential Matrix Module, enabling computations similar to the
Eight-Point algorithm. The resulting mix of visual and positional features is a good inductive bias for pose estimation.

## Installation and Demo

Anaconda install:
```
conda install environment.yml
```
Download & extract pretrained models replicating paper results:
```
wget https://fouheylab.eecs.umich.edu/~cnris/rel_pose/modelcheckpoints/pretrained_models.zip
unzip pretrained_models.zip
```
Demo script to predict pose on arbitrary image:
```
python demo.py --imgs demo/matterport --ckpt pretrained_models/matterport.pth
python demo.py --img demo/interiornet_t --ckpt pretrained_models/interiornet_t.pth
python demo.py --img demo/streetlearn_t --ckpt pretrained_models/streetlearn_t.pth
```

## Evaluation

Download and setup data following the setups of Jin et al. and Cai et al.
```

```
Evaluation scripts are as follows
```
sh scripts/eval_matterport.sh
sh scripts/eval_interiornet.sh
sh scripts/eval_interiornet_t.sh
sh scripts/eval_streetlearn.sh
sh scripts/eval_streetlearn_t.sh
```

## Training

Data setup is the same as in evaluation. Training scripts are as follows:
```
sh scripts/train_matterport.sh
sh scripts/train_interiornet.sh
sh scripts/train_interiornet_t.sh
sh scripts/train_streetlearn.sh
sh scripts/train_streetlearn_t.sh
```


## Citation
If you use this code for your research, please consider citing:
```
@inProceedings{Rockwell2021,
  author = {Chris Rockwell and Justin Johnson and David F. Fouhey},
  title = {The 8-Point Algorithm as an Inductive Bias for Relative Pose Prediction by ViTs},
  booktitle = {3DV},
  year = 2022
}
```

## Special Thanks
Thanks to <a href="https://jinlinyi.github.io/">Linyi Jin</a>, <a href="https://www.cs.cornell.edu/~ruojin/">Ruojin Cai</a> and <a href="https://zachteed.github.io/">Zach Teed</a> for help replicating and building upon their works. Thanks to <a href="https://mbanani.github.io/">Mohamed El Banani</a>, <a href="http://kdexd.xyz/">Karan Desai</a> and <a href="https://nileshkulkarni.github.io/">Nilesh Kulkarni</a> for their many helpful suggestions. Thanks to Laura Fink and UM DCO for their tireless support with computing!
