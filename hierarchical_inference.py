from tensorflow.contrib.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os

from config import experiment_name, base_dir, classes_dict, labels_dict, models_dict, datasets

input_shape = (224, 224, 3)
arch_config = (4,4)
batch_size = 1

val_datagen = ImageDataGenerator(
        rescale=1./255)

nd = {}
for k,v in labels_dict.items():
    nd[v] = k

for model_name in models.keys():
        modelf = models_dict[model_name](input_shape, arch_config[0])
        modelf.load_weights('models/' + model_name + '{}-{}-{}.h5'.format(experiment_name, model_name, dataset_name))
        modelsec = models_dict[model_name](input_shape, arch_config[1])
        modelsec.load_weights('models/' + model_name + '{}-{}tl-{}.h5'.format(experiment_name, model_name, dataset_name))

        report = open('results/{}-{}-{}-{}.txt'.format(experiment_name, model_name, arch_config[0], arch_config[1]), 'w')
        for val_dataset_name in datasets:

            report.write('\n\nEvaluation for the {} dataset\n\n'.format(val_dataset_name))

            test_generator = val_datagen.flow_from_directory(
                    os.path.join(base_dir, '{}/val'.format(val_dataset_name)),
                    target_size=input_shape[:2],
                    classes=classes_dict['all'],
                    batch_size=batch_size)

            expected = []
            predictions = []
            for i in range(len(test_generator.classes) // batch_size):
                item = next(test_generator)
                r = modelf.predict(item[0], batch_size=batch_size, verbose=0)
                pred = np.argmax(r[0])
                if pred in [3,4,5,6]:
                    r = modelsec.predict(item[0], batch_size=batch_size, verbose=0)
                    pred = np.argmax(r[0]) + 3
                if pred == 1:
                    r = modelsec.predict(item[0], batch_size=batch_size, verbose=0)
                    pred = np.argmax(r[0]) + 3
                elif pred != 0:
                    pred -= 1

                expected.extend([np.argmax(x) for x in item[1]])
                predictions.append(pred)

            sess = tf.Session()
            with sess.as_default():
                cm = confusion_matrix(expected, predictions)
                confusion = cm.eval()

            scaled_confusion = confusion.astype(np.float32)
            scaled_confusion /= np.sum(scaled_confusion, axis=1, dtype=np.float32)[:,None]

            prec = []
            report.write('class | prec\n')
            for i, l in enumerate(confusion):
                prec.append(float(l[i])/sum(l))
                report.write('{} | {:.3f}%\n'.format(nd[i], prec[i]))

            report.write('\nConfusion Matrix:\n{}\n\n'.format(confusion))
            mean_acc = np.sum(np.diag(confusion)) / float(np.sum(confusion))
            report.write('Mean Acc: {:.3f}\n'.format(mean_acc))
        report.close()
