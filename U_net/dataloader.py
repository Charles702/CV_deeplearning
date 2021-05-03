import os
from os.path import isdir, exists, abspath, join

import random

import numpy as np
from PIL import Image

class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=.1):
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')

        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]

    def __iter__(self):
        n_train = self.n_train()
        print("train size: ", n_train)

        if self.mode == 'train':
            current = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            endId = len(self.data_files)
       
        while current < endId:
            # todo: load images and labels
            # hint: scale images between 0 and 1
            # hint: if training takes too long or memory overflow, reduce image size!
            im_input = Image.open(self.data_files[current])
            im_label = Image.open(self.label_files[current])    
            
            in_size = 400
            label_size =  ((((((((in_size -4)//2 -4)//2 -4)//2 -4)//2 -4)*2 -4)*2 -4)*2-4)*2 -4
            im_input = im_input.resize((in_size, in_size))
            im_label = im_label.resize((label_size, label_size))
               
            # original 
            data_image = np.array(im_input)/255.
            label_image = np.array(im_label)
            yield (data_image, label_image)            
            
            # flipover
            flip_in = im_input.transpose(Image.FLIP_LEFT_RIGHT) 
            flip_label = im_label.transpose(Image.FLIP_LEFT_RIGHT) 
            data_image = np.array(flip_in)/255.
            label_image = np.array(im_label)
            yield (data_image, label_image)            
            
            #rotate 270
            rotation_in = im_input.transpose(Image.ROTATE_270)
            rotation_label = im_label.transpose(Image.ROTATE_270)
            data_image = np.array(rotation_in)/255.
            label_image = np.array(rotation_label)
            yield (data_image, label_image)
            
            #gamma correction gamma = 0.3
            gamma = 0.3
            data_image =  255.0*(np.array(im_input)/255.0)**(1/gamma)
            data_image =  data_image/255.
            label_image = np.array(im_label)
            yield (data_image, label_image)
            
            current += 1
            

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))