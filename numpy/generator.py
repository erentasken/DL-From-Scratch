import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

import random

from skimage.transform import resize


# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.

        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle

        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        #TODO: implement constructor

        self.curr_epoch = 0
        self.startFlag = False
        self.reset = False

        with open(self.label_path) as f:
            data = json.load(f)
        
        self.items = list(data.items())

    def next(self):
        if self.startFlag:
            self.curr_epoch +=1
        else:
            self.startFlag = True

        start = self.curr_epoch * self.batch_size
        end = start + self.batch_size

        if self.reset:
            self.curr_epoch += 1
            self.reset = False

        if self.shuffle:
            random.shuffle(self.items)

        images, labels = [], []

        # Collect batch images
        for i in range(start, end):
            idx = i % len(self.items)  # Wrap around to fill batch if needed
            k, v = self.items[idx]

            img = np.load(os.path.join(self.file_path, k + ".npy"))
            img = resize(img, self.image_size)
            img = self.augment(img)

            images.append(img)
            labels.append(v)

        if end > len(self.items):
            start = 0
        if end == len(self.items):
            self.curr_epoch = 0
            self.startFlag = False
            self.reset = True

        return np.array(images), np.array(labels)

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        # randomly mirror the image
        if self.mirroring:
            if random.choice([True, False]):
                img = img[:, ::-1, :]
        
        if self.rotation:
            angle = random.choice([90, 180, 270])
            if angle == 90:
                img = np.rot90(img, k=1)
            elif angle == 180:
                img = np.rot90(img, k=2)
            elif angle == 270:
                img = np.rot90(img, k=3)

        return img

    def current_epoch(self):
        # return the current epoch number
        return self.curr_epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function

        return self.class_dict[x]
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method

        try:
            images, labels = self.next()
        except StopIteration:
            print("No more batches available.")
            return

        batch_size = len(images)
        cols = 5
        
        rows = (batch_size + cols) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        axes = axes.flatten() # Flatten the 2D array of axes to 1D for easy iteration

        for i in range(batch_size):
            axes[i].imshow(np.array(images[i]))
            axes[i].set_title(self.class_name(labels[i]))
            axes[i].axis('off')

        plt.show()


