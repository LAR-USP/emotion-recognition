from architectures.mobilenet import generate_mobilenet, generate_mobilenettl
from architectures.inception_v3 import generate_inception, generate_inceptiontl

base_dir = 'datasets'
datasets = ['GLOBAL', 'FERPLUS', 'RAF', 'QIDER']


# Classes used in each scenario, the ordered classes array guarantees the training order
ordered_classes = ['first', 'second', 'all']
classes_dict = {'first': ['happy', 'negative', 'neutral', 'surprise'],
           'second': ['fear', 'disgust', 'angry', 'sad'],
           'all': ['happy', 'neutral', 'surprise', 'fear', 'disgust', 'angry', 'sad']}

labels_dict = {'happy': 0, 'neutral': 1, 'surprise': 2, 'fear': 3, 'disgust': 4, 'angry': 5, 'sad': 6}

ordered_models = ['inception', 'inceptiontl', 'mobilenet', 'mobilenettl']
models_dict = {
        'inception': generate_inception,
        'inceptiontl': generate_inceptiontl,
        'mobilenet': generate_mobilenet,
        'mobilenettl': generate_mobilenettl,
        'lettuce': generate_baseline
}
