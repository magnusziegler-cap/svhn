#This script is an PoC example showing the dataloader-class successfully loads data from disk in the requested way.
import numpy as np
from dataloader import dataloader
import augmentations
import matplotlib.pyplot as plt

# use the augmentations
transforms = augmentations.compose(
    augmentations.grayscale,
    augmentations.shift,
    augmentations.rotate,
    probabilities=[1, 1, 0.5],
    verbose=True
    )

batch_size = 20 #if you dont have batch size 20, you can't hit lucky number 19.
path_to_data = 'training_data\\'

# create Training and Validation dataloaders
training_loader = dataloader(batch_size=batch_size, shuffle=True, from_disk=True, path_to_data=path_to_data, augmentations=transforms)
training_loader.generate_batches()
training_loader.printSettings()
print(f'The Validation set has: {training_loader.num_examples} examples')

# prototypical training loop
num_epochs=1
for e in range(num_epochs):
    print(f'Epoch {e}')

    for batch_images_train, batch_labels_train in training_loader.get_batch():
        # batch_images_train = dataloader.apply_augmentations(batch_images_train, transforms) #alt feature application method

        # prove dataloader and augmentation feature works
        print(batch_images_train[19])
        print(batch_labels_train[19])

        # do some ML things typically, but today just print some images
    
        fig = plt.figure(figsize=(batch_size+2,1+batch_size//2))

        for r in range(0,batch_size):
            image = batch_images_train[r]
            
            plt.subplot(1,batch_size,r+1)
            plt.imshow(image) #get back from pytorch ordering to something plt likes
        plt.show()

        break 

    