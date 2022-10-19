import numpy as np
import scipy.io
import os
from PIL import Image

training_data = scipy.io.loadmat(file_name='data/train_32x32.mat')
print(type(training_data), len(training_data))
print(training_data.keys())
print(training_data['X'].shape)
print(training_data['y'].shape)
X = training_data['X']
y = training_data['y'] 
total_export = 1000
for n in range(0,total_export):
    img = X[:,:,:,n]
    label = y[n]
    out = Image.fromarray(img, mode='RGB')
    path = f'C:\\Users\\maziegle\\Downloads\\svhn\\training_data\\img_{n}_{label[0]}.png'
    out.save(path)
