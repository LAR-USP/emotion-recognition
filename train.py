from tensorflow.contrib.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os

from utils.keras_to_tensorflow import h5_to_pb
from utils.imagenet_utils import preprocess_input

from config import datasets, experiment_name, base_dir, ordered_classes, classes_dict, ordered_models, models_dict

def init_generators(input_shape, batch_size, dataset_name, model_name, t, weighted=True):
    '''
    Creates image data generators for training and defines class weights to compensate class imbalance (optional)
    '''
    train_generator = train_datagen.flow_from_directory(
            os.path.join(base_dir, '{}/train'.format(dataset_name)),  # this is the target directory
            target_size=input_shape[:2],  # all images will be resized to 150x150
            classes=classes_dict[t],
            batch_size=batch_size)

    val_generator = val_datagen.flow_from_directory(
            os.path.join(base_dir,'{}/val'.format(dataset_name)),  # this is the target directory
            target_size=input_shape[:2],  # all images will be resized to 150x150
            classes=classes_dict[t],
            batch_size=batch_size)

    dataset_size = len(train_generator.classes)
    dataset_distribution = np.bincount(train_generator.classes)
    c_weights = float(dataset_size)/(len(dataset_distribution)*dataset_distribution)

    class_weight_dict = {}
    if weighted:
        for i, w in enumerate(c_weights):
            if w > 2:
                w = 2.0
            class_weight_dict[i] = w

    return train_generator, val_generator, class_weight_dict

def test(report, input_shape, batch_size, val_dataset_name, labels_dict, model, model_name, t):
    '''
        Creates a report file with testing metrics
    '''
    report.write('\n\nEvaluation for the {} dataset\n\n'.format(val_dataset_name))

    test_generator = val_datagen.flow_from_directory(
            os.path.join(base_dir, '{}/val'.format(val_dataset_name)),
            target_size=input_shape[:2],
            classes=classes[t],
            batch_size=batch_size)

    nd = {}
    for k,v in labels_dict.items():
        nd[v] = k

    expected = []
    predictions = []
    for i in range(len(test_generator.classes) // batch_size):
        batch = next(test_generator)
        # r = model.evaluate(batch[0], batch[1], batch_size=batch_size, verbose=0)
        r = model.predict(batch[0], batch_size=batch_size, verbose=0)
        expected.extend([np.argmax(x) for x in batch[1]])
        predictions.extend([np.argmax(x) for x in r])

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
        # print('Class: {} Acc: {}%'.format(nd[i], acc[i]))
        report.write('{} | {:.3f}%\n'.format(nd[i], prec[i]))

    # print('Confusion Matrix:\n{}\n'.format(confusion))
    report.write('\nConfusion Matrix:\n{}\n\n'.format(confusion))
    # report.write('\nScaled Confusion Matrix:\n{}\n\n'.format(scaled_confusion))

    true_positives = np.diag(confusion)
    false_negatives = np.array([np.sum(l) - true_positives[i]
        for i, l in enumerate(confusion)])
    false_positives = np.array([np.sum(c) - true_positives[i]
        for i, c in enumerate(confusion.T)])
    rho = true_positives / (true_positives + false_negatives)
    pi = true_positives / (true_positives + false_positives)
    f1 = (2 * pi * rho) / (pi + rho)
    report.write('F1: {}\n'.format(f1))
    f1_macro = np.mean(f1)
    report.write('F1 Macro: {:.3f}\n'.format(f1_macro))
    # print('Mean Acc: {}'.format(np.mean(acc)))
    mean_acc = np.sum(np.diag(confusion)) / float(np.sum(confusion))
    report.write('Mean Acc: {:.3f}\n'.format(mean_acc))
    return mean_acc

input_shape = (224, 224, 3)
batch_size = 32
num_classes = 7

train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

val_datagen = ImageDataGenerator(
        rescale=1./255)

overall_report = open('results/{}.txt'.format(experiment_name), 'w')
overall_report.write('model-name,trained-on,evaluated-on,accuracy\n')

for model_name in ordered_models:
    for dataset_name in datasets:
        for t in ordered_classes:
            num_classes = len(classes_dict[t])
            train_generator, val_generator, class_weights = init_generators(input_shape,
                                                                batch_size,
                                                                dataset_name,
                                                                model_name,
                                                                t)
            dataset_size = len(train_generator.classes)
            validation_size = len(val_generator.classes)
            labels_dict = train_generator.class_indices

            model = models_dict[model_name](input_shape, num_classes)
            if t == 'second':
                model.load_weights('models/{}-{}-{}.h5'.format(experiment_name, model_name.replace('tl',''), dataset_name))

            model.fit_generator(
                    train_generator,
                    steps_per_epoch=dataset_size // batch_size,
                    validation_data=val_generator,
                    validation_steps=validation_size // batch_size,
                    epochs=100)

            model.save_weights('models/{}-{}-{}-{}.h5'.format(experiment_name, model_name, dataset_name, t))

            report = open('results/{}-{}-{}-{}.txt'.format(experiment_name, model_name, dataset_name, t), 'w')
            for val_dataset_name in datasets:
                acc = test(report, input_shape, batch_size,
                           val_dataset_name, labels_dict, model, model_name, t)
                overall_report.write('{},{},{},{:.3f}\n'.format(model_name,dataset_name,val_dataset_name,acc))
            report.close()

            # ''' Exporting protobuf '''
            h5_to_pb(model, 'models/{}-{}-{}.pb'.format(experiment_name, model_name, dataset_name))
overall_report.close()
