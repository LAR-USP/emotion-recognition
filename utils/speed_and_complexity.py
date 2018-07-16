from inception_v3 import generate_inceptiontl
from mobilenet import generate_mobilenettl
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input
import numpy as np
import os
import timeit

base_dir = '../datasets/'
input_shape = (224,224,3)

val_datagen = ImageDataGenerator(
        rescale=1./255)

test_generator = val_datagen.flow_from_directory(
        os.path.join(base_dir, 'complexity'),
        target_size=input_shape[:2],
        batch_size=1)

# inception = generate_inceptiontl(input_shape, 7)
mobilenet = generate_mobilenettl(input_shape, 7)
# model = Model(inputs=inception.input, outputs=inception.layers[172].output)
model = Model(inputs=mobilenet.input, outputs=mobilenet.layers[-5].output)

speeds1 = []
speeds2 = []

for i in range(len(test_generator.classes)):
    item = next(test_generator)
    start_time = timeit.default_timer()
    mobilenet.predict(item[0], batch_size=1, verbose=0)
    end_time = timeit.default_timer()
    speeds1.append(end_time - start_time)

for i in range(len(test_generator.classes)):
    item = next(test_generator)
    start_time = timeit.default_timer()
    model.predict(item[0], batch_size=1, verbose=0)
    end_time = timeit.default_timer()
    speeds2.append(end_time - start_time)


speeds1 = np.array(speeds1)[1:]
speeds2 = np.array(speeds2)[1:]

print('SPEED INSPECTION')
print(speeds1)
# print(np.mean(speeds1))
# print(np.std(speeds1))

# print('SPEED 2')
# print(np.mean(speeds1 - speeds2))
# print(np.std(speeds1 - speeds2))

print('SPEED INSPECTION H')
final = 2*speeds1 - speeds2
print(final)
print(np.mean(final))
print(np.std(final))
