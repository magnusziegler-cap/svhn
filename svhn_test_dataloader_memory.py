#This script is an PoC example showing the dataloader-class successfully loads data from memory in the requested way.
import numpy as np
import scipy.io
from dataloader import dataloader
import augmentations
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# directly load the dataset into memory
training_data = scipy.io.loadmat(file_name='data/train_32x32.mat')
X = training_data['X']
y = training_data['y'] 
print(f'Input Dataset has {len(y)} examples and labels.')

# split "Testing Data" into training, validation(aka dev) sets with 70/30 split, transpose y to fit sklearn [HWCN]
X_train, X_validation, y_train, y_validation = train_test_split(np.transpose(X,axes=(3,2,0,1)), y, random_state=19, train_size=0.7)

# use the augmentations
transforms = augmentations.compose(
    augmentations.grayscale,
    augmentations.shift,
    augmentations.rotate,
    probabilities=[1, 1, 0.5],
    verbose=True
    )

batch_size = 20

# create Training and Validation dataloaders
training_loader = dataloader(batch_size=batch_size, shuffle=True, augmentations=transforms)
training_loader.load_direct(X_train,y_train)
training_loader.generate_batches()
training_loader.printSettings()
print(f'The Validation set has: {training_loader.num_examples} examples')

validation_loader = dataloader(batch_size=batch_size, shuffle=True, augmentations=transforms)
validation_loader.load_direct(X_validation,y_validation)
validation_loader.generate_batches()
validation_loader.printSettings()
print(f'The Validation set has: {validation_loader.num_examples} examples')

# training loop
num_epochs=1
for e in range(num_epochs):
    print(f'Epoch {e}')

    for batch_images_train, batch_labels_train in training_loader.get_batch():
        # batch_images_train = dataloader.apply_augmentations(batch_images_train, transforms) #alternate transform application style
        
        # prove dataloader and augmentation logic works
        print(batch_images_train[19])
        print(batch_labels_train[19])
        
        # do some ML things
        fig = plt.figure(figsize=(batch_size+2,1+batch_size//2))

        for r in range(0,batch_size):
            image = batch_images_train[r]
            
            plt.subplot(1,batch_size,r+1)
            plt.imshow(image) 
        plt.show()

        break
    