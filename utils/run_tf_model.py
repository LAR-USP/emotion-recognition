from tensorflow.core.framework import graph_pb2
import tensorflow as tf
import numpy as np
import cv2

graph_path = '../models/mobilenet-fer2013.pb'
sess = tf.Session()
with open(graph_path, "rb") as f:
    output_graph_def = graph_pb2.GraphDef()
    output_graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(output_graph_def, name="")

a.set_learning_phase(0)

x = sess.graph.get_tensor_by_name('input_1:0')
c = sess.graph.get_tensor_by_name('conv1_bn/keras_learning_phase:0')
y = sess.graph.get_tensor_by_name('output_node:0')

img = cv2.imread('test.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (48,48))
imgs = np.expand_dims(img, 0)

new_scores = sess.run(y, feed_dict={x: imgs, c: False})
print(np.argmax(new_scores))
