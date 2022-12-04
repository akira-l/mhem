# ELP 

## Introduction

This project is an implementation of [*Penalizing the Hard Example But Not Too Much: A Strong Baseline for Fine-Grained Visual Classification*]

## Insight 

* Proper hard example mining boost FGVC performance. Only by modulating the loss function, a naive ResNet-50 baseline can outperform many complex models. 

## Requirements

Python 3 & Pytorch >= 0.4.0 

## Datasets Orgnization 

Similar to [DCL](https://github.com/JDAI-CV/DCL). 

## Training

Run `train.py` to train ELP.

For CUB / STCAR / AIR 

```shell
python train.py --data $DATASET --epoch 360 --backbone resnet50 \
                    --tb 16 --tnw 16 --vb 512 --vnw 16 \
                    --lr 0.0008 --lr_step 60 \
                    --cls_lr_ratio 10 --start_epoch 0 \
                    --detail training_descibe --size 512 \
                    --crop 448 
```

You can rewrite line 98-125 in utils/train_model.py for your own codebase. 

## Citation
Please cite MHEM paper if you find MHEM is helpful in your work:
```
@ARTICLE{9956020,
  author={Liang, Yuanzhi and Zhu, Linchao and Wang, Xiaohan and Yang, Yi},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Penalizing the Hard Example But Not Too Much: A Strong Baseline for Fine-Grained Visual Classification}, 
  year={2022},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TNNLS.2022.3213563}}
```
