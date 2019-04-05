import tensorflow as tf
from tensorflow.python.platform import gfile

from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim


graph_pb = 'inception_v4_graph.pb'

num_classes = 100


def main(_):
    with tf.Graph().as_default() as graph:
        network_fn = nets_factory.get_network_fn(
            'inception_v4',
            num_classes=num_classes,
            is_training=False
        )

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

        graph_def = graph.as_graph_def()
        with gfile.GFile(graph_pb, 'wb') as f:
            f.write(graph_def.SerializeToString())


if __name__ == '__main__':
    tf.app.run()
