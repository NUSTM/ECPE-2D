
# ECPE-2D: Emotion-Cause Pair Extraction based on Joint Two-Dimensional Representation, Interaction and Prediction

This repository contains the code for our ACL 2020 paper:


Zixiang Ding, Rui Xia, Jianfei Yu. ECPE-2D: Emotion-Cause Pair Extraction based on Joint Two-Dimensional Representation, Interaction and Prediction. ACL 2020. [[pdf](https://www.aclweb.org/anthology/2020.acl-main.288.pdf)]

Please cite our paper if you use this code.

## Dependencies

- **Python 2** (tested on python 2.7.15)
- [Tensorflow](https://github.com/tensorflow/tensorflow) 1.13.1
- [BERT-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

## Usage

- For ECPE-2D(Indep, Inter-CE, Inter-EC) models, run:
    - python main.py


- For ECPE-2D(Inter-EC(BERT)) models, run:
    - python Bert_main.py
