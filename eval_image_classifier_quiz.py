# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
#
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset.
cd /home/sophiayu/tensorflow/quiz-w7-2
python eval_image_classifier_quiz.py \
  --checkpoint_path=/home/sophiayu/Models/train_ckpt \
  --eval_dir=~ \
  --dataset_name=quiz \
  --dataset_split_name=test \
  --dataset_dir=/home/sophiayu/Data/手写汉字识别数据/test2 \
  --model_name=inception_v4 \
  --preprocessing_name=inception \
  --batch_size=32 \
  --max_num_batches=128"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf
import os
import pandas as pd
import numpy as np

from datasets import dataset_factory
from preprocessing import inception_preprocessing
from nets import nets_factory
from preprocessing import preprocessing_factory
from PIL import Image

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 50, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS

is_training = False
preprocessing_name = FLAGS.model_name

graph = tf.Graph().as_default()

image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name, is_training)

network_fn = nets_factory.get_network_fn(FLAGS.model_name, num_classes=100, is_training=is_training)

placeholder = tf.placeholder(name='input', dtype=tf.string)
image = tf.image.decode_jpeg(placeholder,channels=3)
image = image_preprocessing_fn(image, 320, 320)
image = tf.expand_dims(image, 0)
logit, _=network_fn(image)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, FLAGS.checkpoint_path)

filenames=[]
recall5word = []

idToword={0:'且',1:'世',2:'东',3:'九',4:'亭',5:'今',6:'从',7:'令',8:'作',9:'使',10:'侯',11:'元',12:'光',13:'利',14:'印',15:'去',16:'受',17:'右',18:'司',19:'合',20:'名',21:'周',22:'命',23:'和'
,24:'唯',25:'堂',26:'士',27:'多',28:'夜',29:'奉',30:'女',31:'好',32:'始',33:'字',34:'孝',35:'守',36:'宗',37:'官',38:'定',39:'宜',40:'室',41:'家',42:'寒',43:'左'
,44:'常',45:'建',46:'徐',47:'御',48:'必',49:'思',50:'意',51:'我',52:'敬',53:'新',54:'易',55:'春',56:'更',57:'朝',58:'李'
,59:'来',60:'林',61:'正',62:'武'
,63:'氏'
,64:'永'
,65:'流'
,66:'海'
,67:'深'
,68:'清'
,69:'游'
,70:'父'
,71:'物'
,72:'玉'
,73:'用'
,74:'申'
,75:'白'
,76:'皇'
,77:'益'
,78:'福'
,79:'秋'
,80:'立'
,81:'章'
,82:'老'
,83:'臣'
,84:'良'
,85:'莫'
,86:'虎'
,87:'衣'
,88:'西'
,89:'起'
,90:'足'
,91:'身'
,92:'通',93:'遂'
,94:'重'
,95:'陵'
,96:'雨'
,97:'高'
,98:'黄'
,99:'鼎'}
n = 0
for filename in os.listdir(FLAGS.dataset_dir):
    filenames.append(filename)
    filepath = os.path.join(FLAGS.dataset_dir, filename)
    image_value = open(filepath, 'rb').read()
    logit_value = sess.run([logit], feed_dict={placeholder:image_value})
    logit_value = np.squeeze(logit_value)
    recall5 = ''
    for j in range(5):
        id = np.argmax([logit_value])
        word = idToword[id]
        recall5 += word
        logit_value[id] = 0
    recall5word.append(recall5)
    n += 1
    if n % 100 == 0:
        print(n)
        print(recall5)
        #break

export_data = {'filename': filenames, 'label': recall5word}
# print(export_data)
exportt = pd.DataFrame(export_data)
exportt.to_csv("/home/dl/offline/comp.jpg/export_data.csv", index=False)

# def main(_):
#   if not FLAGS.dataset_dir:
#     raise ValueError('You must supply the dataset directory with --dataset_dir')
#
#   tf.logging.set_verbosity(tf.logging.INFO)
#   with tf.Graph().as_default():
#     tf_global_step = slim.get_or_create_global_step()
#
#     ######################
#     # Select the dataset #
#     ######################
#     print('========================== Select the dataset =====================')
#     dataset = []
#     widthlist = []
#     hightlist = []
#     for filename in os.listdir(FLAGS.dataset_dir):
#         path = os.path.join(FLAGS.dataset_dir, filename)
#         image_data = tf.gfile.FastGFile(path, 'rb').read()
#         image_data = tf.image.decode_jpeg(image_data, channels=3)
#         image_data = tf.image.convert_image_dtype(image_data, dtype=tf.uint8)
#         img = Image.open(path)
#         sizep = img.size
#         dataset.append(image_data)
#         widthlist.append(sizep[0])
#         hightlist.append(sizep[1])
#         if len(dataset) >= 96:
#             break
#
#     # dataset = dataset_factory.get_dataset(
#     #     FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
#
#     ####################
#     # Select the model #
#     ####################
#     print('========================== Select the model =====================')
#     network_fn = nets_factory.get_network_fn(
#         FLAGS.model_name,
#         num_classes=100,
#         is_training=False)
#
#     ##############################################################
#     # Create a dataset provider that loads data from the dataset #
#     ##############################################################
#     # provider = slim.dataset_data_provider.DatasetDataProvider(
#     #     dataset,
#     #     shuffle=False,
#     #     common_queue_capacity=2 * FLAGS.batch_size,
#     #     common_queue_min=FLAGS.batch_size)
#     # # [image, label] = provider.get(['image', 'label'])
#     # image = provider.get(['image'])
#     # label -= FLAGS.labels_offset
#
#     #####################################
#     # Select the preprocessing function #
#     #####################################
#     # preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
#     # image_preprocessing_fn = preprocessing_factory.get_preprocessing(
#     #     preprocessing_name,
#     #     is_training=False)
#
#     eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size
#
#     print('========================== 数据预处理 =====================')
#     numdata = len(dataset)
#     for i in range(numdata):
#         img = dataset[i]
#         img = tf.image.convert_image_dtype(img, dtype=tf.float32)
#         # Crop the central region of the image with an area containing 87.5% of
#         # the original image.中央裁切
#         img = tf.image.central_crop(img, central_fraction=0.875)
#
#         # Resize the image to the specified height and width.
#         img = tf.expand_dims(img, 0)
#         img = tf.image.resize_bilinear(img, [320, 320],
#                                          align_corners=False)
#         img = tf.squeeze(img, [0])
#         img = tf.subtract(img, 0.5)
#         img = tf.multiply(img, 2.0)
#         dataset[i] = img
#         # dataset[i] = inception_preprocessing.preprocess_for_eval(dataset[i], widthlist[i], hightlist[i], central_fraction=1)
#     #
#     print('========================== 分块 =====================')
#     images = []
#     temp = 0
#     while temp <= numdata:
#         if temp+32 <= numdata:
#             images.append(dataset[temp:temp+32])
#         elif temp+32 <= numdata+32:
#             # images.append(dataset[temp:])
#             break
#         temp += 32
#     # image = dataset[0]
#     # images = tf.train.batch(
#     #     [image],
#     #     batch_size=FLAGS.batch_size,
#     #     num_threads=1,
#     #     capacity=5 * FLAGS.batch_size)
#
#     ####################
#     # Define the model #
#     ####################
#     print('========================== 预测 =====================')
#     # print(images.shape)
#     placehodler = tf.placeholder(name='input', dtype=tf.float32)
#     # logits, _ = network_fn(images[0])
#     immm = dataset[0]
#     immm = tf.expand_dims(immm, 0)
#     logits, _ = network_fn(immm)
#
#     saver = tf.train.Saver()
#     sess = tf.Session()
#     # checkpoint_path = tf.train.checkpoint(FLAGS.checkpoint_path)
#     # print('checkpoint_path', checkpoint_path)
#     saver.restore(sess, FLAGS.checkpoint_path)
#
#     # numbatch = len(images)
#     # for i in range(numbatch):
#     #     image_value = images[i]
#     #     print('image_value size', image_value.shape)
#     #     # image_value = dataset[i]
#     #     logit_value = sess.run([logits], feed_dict={placehodler:image_value})
#
#     for i in range(numdata):
#         image_value = dataset[i]
#         logit_value = sess.run([logits], feed_dict={placehodler: image_value})
#
#     print(logit_value)
#     # print(np.argmax(logit_value))
#
#     print('========================== 完成预测 =====================')
#
#     # if FLAGS.moving_average_decay:
#     #   variable_averages = tf.train.ExponentialMovingAverage(
#     #       FLAGS.moving_average_decay, tf_global_step)
#     #   variables_to_restore = variable_averages.variables_to_restore(
#     #       slim.get_model_variables())
#     #   variables_to_restore[tf_global_step.op.name] = tf_global_step
#     # else:
#     #   variables_to_restore = slim.get_variables_to_restore()
#
#     # predictions = tf.argmax(logits, 1)
#     # labels = tf.squeeze(labels)
#
#     # Define the metrics:
#     # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
#     #     'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
#     #     'Recall_5': slim.metrics.streaming_recall_at_k(
#     #         logits, labels, 5),
#
#     # print('====================== logits =================\n', logits[0:10])
#     # print(logits)
#     # out_df1 = pd.DataFrame(logits)
#     # out_df1.columns = ["filename", "lable"]
#     #
#     # out_df = out_df1
#     # out_df.to_csv("/home/sophiayu/Data/手写汉字识别数据/logits.csv", index=False)
#
#     # })
#
#     # Print the summaries to screen.
#     # for name, value in names_to_values.items():
#     #   summary_name = 'eval/%s' % name
#     #   op = tf.summary.scalar(summary_name, value, collections=[])
#     #   op = tf.Print(op, [value], summary_name)
#     #   tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
#
#     # # TODO(sguada) use num_epochs=1
#     # if FLAGS.max_num_batches:
#     #   num_batches = FLAGS.max_num_batches
#     # else:
#     #   # This ensures that we make a single pass over all of the data.
#     #   num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))
#     #
#     # if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
#     #   checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
#     # else:
#     #   checkpoint_path = FLAGS.checkpoint_path
#     #
#     # tf.logging.info('Evaluating %s' % checkpoint_path)
#     #
#     # slim.evaluation.evaluate_once(
#     #     master=FLAGS.master,
#     #     checkpoint_path=checkpoint_path,
#     #     logdir=FLAGS.eval_dir,
#     #     num_evals=num_batches,
#     #     variables_to_restore=variables_to_restore)


# if __name__ == '__main__':
#   tf.app.run()
