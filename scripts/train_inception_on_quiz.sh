#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Cifar10 dataset
# 2. Trains a CifarNet model on the Cifar10 training set.
# 3. Evaluates the model on the Cifar10 testing set.
#
# Usage:
# cd slim
# ./scripts/train_cifarnet_on_cifar10.sh
set -e

# Where the checkpoint and logs will be saved to.
# TRAIN_DIR=~/tmp/quiz-model

# Where the dataset is saved to.
# DATASET_DIR=~/tmp/quiz

# Download the dataset
# python download_and_convert_data.py \
#   --dataset_name=quiz \
#   --dataset_dir=${DATASET_DIR}

# Run training.
# train_dir 保存checkpoin路径
# dataset_name 枚举值 4个默认公开数据集
# dataset_split_name  指定train还是validation
# model_name 网络名称，在nets目录下查找
# preprocessing_name 预处理名称
# max_number_of_steps 最大训练步数
# batch_size 每批次图片数量
# save_interval_secs  每隔多少秒保存一个checkpoint
# save_summaries_secs  每隔多少秒写summaries
# log_every_n_steps 每隔多少个step输出一次log
# optimizer 优化器 sgd(随机梯度下降) rmsprop
# learning_rate 学习速率
# learning_rate_decay_factor 学习率衰减因子
# num_epochs_per_decay 经过多少epoch衰减一次学习速率
# weight_decay L2正则化中衰减因子
# checkpoint_exclude_scopes 需要重新训练的层
python train_image_classifier.py \
  --train_dir=/home/sophiayu/Models/quiz-model/train_ckpt \
  --dataset_name=quiz \
  --dataset_split_name=train \
  --dataset_dir=/home/sophiayu/Data/手写汉字识别数据/train \
  --checkpoint_path=/home/sophiayu/Models/inception_v4/inception_v4.ckpt\
  --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits\
  --model_name=inception_v4 \
  --preprocessing_name=inception \
  --max_number_of_steps=100000 \
  --batch_size=32 \
  --save_interval_secs=300 \
  --save_summaries_secs=300 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --learning_rate=0.01 \
  --learning_rate_decay_factor=0.1 \
  --num_epochs_per_decay=200 \
  --weight_decay=0.004

# train集验证：
# eval_dir  验证结果输出
python3 eval_image_classifier.py
  --checkpoint_path=/home/sophiayu/Models/quiz-model/train_ckpt\
  --eval_dir=/path/to/train_eval\
  --dataset_name=quiz\
  --dataset_split_name=train\
  --dataset_dir=/home/sophiayu/Data/手写汉字识别数据/train\
  --model_name=inception_v4\
  --preprocessing_name=inception \
  --batch_size=32\
  --max_num_batches=128

# Run evaluation.
# eval_dir  验证结果输出
python eval_image_classifier.py \
  --checkpoint_path=/home/sophiayu/Models/quiz-model/train_ckpt \
  --eval_dir=/home/sophiayu/Models/quiz-model/train_ckpt \
  --dataset_name=quiz \
  --dataset_split_name=validation \
  --dataset_dir=/home/sophiayu/Data/手写汉字识别数据/train \
  --model_name=inception_v4\
  --preprocessing_name=inception \
  --batch_size=32\
  --max_num_batches=128

# 统一脚本：
python3 train_eval_image_classifier.py
  --dataset_name=quiz
  --dataset_dir=/home/sophiayu/Data/手写汉字识别数据/train\
  --checkpoint_path=/home/sophiayu/Models/quiz-model/train_ckpt\
  --model_name=inception_v4\
  --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits\
  --optimizer=rmsprop\
  --train_dir=/home/sophiayu/Models/quiz-model/train_ckpt\
  --learning_rate=0.001\
  --dataset_split_name=validation\
  --eval_dir=/path/to/eval\
  --max_num_batches=128
