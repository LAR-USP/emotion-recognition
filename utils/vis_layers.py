from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from imagenet_utils import preprocess_input
from mobilenet import generate_mobilenet
from slimception import generate_slimception
from inception_v3 import generate_inception
from matplotlib import pyplot as plt
from keras import backend as K
import numpy as np
import cv2
import os

input_shape = (224, 224, 3)
num_classes = 7
# weights_path = '../models/mobilenet-monster-noweight.h5'
weights_path = '../models/inception-monster-noweight.h5'
img_path = 'sample/sad.png'

# model = generate_mobilenet(input_shape, num_classes)
slimodel = generate_slimception(input_shape, num_classes)
model = generate_inception(input_shape, num_classes)
model.load_weights(weights_path)
# print(slimodel.count_params())
l = model.get_layer('mixed5')
inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
outputs = [l.output]          # all layer outputs
functor = K.function([inp]+ [K.learning_phase()], outputs) # evaluation function

img = cv2.resize(cv2.imread(img_path), (input_shape[:2])) / 255
data = np.expand_dims(img, axis=0)

# Testing
# 0 for test mode, no normalization or dropout
layer_outs = functor([data, 0.])
layer_activation = layer_outs[0][0]
print('SHAPE OF SELECTED LAYER: {}'.format(layer_activation.shape))
activations = np.transpose(layer_activation, (2, 0, 1))

n = int(np.ceil(np.sqrt(activations.shape[0])))
fig = plt.figure(figsize=(24,16))
for i in range(len(activations)):
    ax = fig.add_subplot(n,n,i+1)
    x = activations[i].copy()
    cv2.normalize(x, x, 0, 255, cv2.NORM_MINMAX)
    ax.imshow(x, cmap='gray')
plt.show()
# activations = np.hstack(activations)
# plt.imshow(activations, interpolation='None', cmap='gray')
# plt.show()
