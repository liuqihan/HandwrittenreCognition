import numpy as np
import tensorflow as tf

model_pb = 'frozen_inception_v4.pb'
label_file = '../comp.jpg/train/labels.txt'
output_tensor_name = 'prediction'

with open(model_pb, 'rb') as f:
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default() as g:
        graph_def = graph.as_graph_def()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


with open(label_file) as inf:
    labels = list(inf)

labels = np.array([l.strip().split(':')[1] for l in labels])


sess = tf.Session(graph=graph)
predictions = sess.graph.get_tensor_by_name(output_tensor_name + ':0')

with open('/home/dl/offline/comp.jpg/test2/d8a603761031e62f0f78db989bab3c4e8cc4dfd9.jpg', 'rb') as inf:
    image_data = inf.read()

pred = sess.run(predictions, {'DecodeJpeg/contents:0': image_data})[0]

idx = np.argsort(pred)[-5:]
for i in idx:
    print("{}:{:.2}%" .format(labels[i], pred[i] * 100))
