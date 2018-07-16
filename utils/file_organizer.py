import os
from random import shuffle

# for d in os.listdir('datasets/emotions'):
#         if d == 'train' or d == 'test':
#             continue
#         os.mkdir('datasets/emotions/' + 'train/' + d)
#         os.mkdir('datasets/emotions/' + 'test/' + d)
#         images = ['datasets/emotions/'+ d + '/' + sd
#                  for sd in os.listdir('datasets/emotions/' + d)]
#         shuffle(images)
#         pct = int(0.75*len(images))
#         for i in images[:pct]:
#             os.rename(i, 'datasets/emotions/train' + '/' + d + '/' + i.split('/')[-1])
#             print(i, 'datasets/emotions/train' + '/' + d + '/' + i.split('/')[-1])
#
#         for i in images[pct:]:
#             print(i, 'datasets/emotions/test' + '/' + d + '/' + i.split('/')[-1])
#             os.rename(i, 'datasets/emotions/test' + '/' + d + '/' + i.split('/')[-1])

for d in os.listdir('datasets/emotions'):
    for sd in os.listdir('datasets/emotions/' + d):
        v = 'datasets/emotions/'+ d + '/' + sd
        for ssd in os.listdir(v):
            file_name = sd + '/' + ssd
            for img in os.listdir('datasets/emotions/'+d+'/'+file_name):
                name = 'datasets/emotions/' + d + '/' + '_'.join(file_name.split()) + '_' + img
                original = v + '/' + ssd + '/' + img
                print(original, name)
                os.rename(original, name)
