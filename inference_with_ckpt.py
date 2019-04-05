import os

import numpy as np
import tensorflow as tf

from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim


# Load the model
sess = tf.Session()

num_classes = 100

# Select the model #
####################
network_fn = nets_factory.get_network_fn(
    'inception_v4',
    num_classes=num_classes,
    is_training=False
)

ckpt_filename = '/home/dl/offline/train_ckpt/model.ckpt-4657'

src_dir = '/home/dl/offline/comp.jpg/test2'

label_file = '/home/dl/offline/comp.jpg/train/labels.txt'

results_csv =  '/home/dl/offline/results.csv'


def eval():
    preprocessing_name = 'inception_v4'
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)
    input_tensor = tf.placeholder(tf.string, name='DecodeJpeg/contents')
    image = tf.image.decode_jpeg(input_tensor, channels=3)
    image = image_preprocessing_fn(image, 299, 299)
    image = tf.expand_dims(image, 0)
    logits, end_points = network_fn(image)
    prediction = tf.nn.softmax(logits, name='prediction')
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_filename)
    print('restored from file %s' % ckpt_filename)
    return logits, prediction, end_points


with open(label_file) as inf:
    labels = list(inf)

labels = np.array([l.strip().split(':')[1] for l in labels])

logits, prediction, end_points = eval()


with open(results_csv, 'w') as csv:
    csv.write('filename,label\n')
    for r,ds,fs in os.walk(src_dir):
        for f in fs:
            if f.endswith('.jpg'):
                print('handle [{0}]'.format(f))
                image_name = os.path.join(r, f)
                with open(image_name, 'rb') as inf:
                    image_data = inf.read()
                logit_values, predict_values, end_points_values = sess.run(
                    [logits, prediction, end_points],
                    feed_dict={'DecodeJpeg/contents:0': image_data})
                print(np.shape(logit_values))

                pred = predict_values[0, :]
                idx = np.argsort(pred)[-5:]
                print(pred[idx])
                print(idx)
                print(labels[idx])

                csv.write('{},{}\n'.format(f, ''.join(labels[idx]) ))
