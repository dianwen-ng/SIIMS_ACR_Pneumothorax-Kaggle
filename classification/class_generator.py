import numpy as np
import random
import os
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import keras

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, datacsv, 
                 image_size=512,
                 batch_size=32,
                 phase = 'train',
                 bounding_box=None,
                 shuffle=True,
                 num_ensemble=None,
                 seed = 612):
        
        np.random.seed(seed)
        
        self.datacsv = datacsv
        self.target = datacsv.set_index('Filename').to_dict()['Label']
        self.image_size = image_size
        self.batch_size = batch_size
        self.phase = phase
        self.shuffle = shuffle
        self.num_ensemble = num_ensemble
        
        # convert bounding box csv to hash table if provided 
        if bounding_box is not None:
            self.bounding_box = bounding_box.set_index('Filename').to_dict()
            
        else:
            self.bounding_box = bounding_box


        self.on_epoch_end()
    
    
    def augmentation(self, image,
                     phase = 'train'):
        
        ## augmentation for training
        if phase == 'train':
            
            # random horizontal flip
            if random.randint(0,1):
                image = cv2.flip(image, 1)
            
            # random brightness:
            factor = np.random.choice([1, 1.25, 1.5, 2], p = (0.30, 0.35, 0.25, 0.1))
            factor = 1.0 / factor
            table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype("uint8")
            image = cv2.LUT(image, table)
            
            # random rotation
            if random.randint(0, 1):
                angle = random.gauss(mu=0.0, sigma=2.0)
                M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, 1.0)
                image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            
            # random histogram equalizer
            if random.randint(0,1):
                img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
                img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
                image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB) # to RGB
                
            return image
        
        ## augmentation for testing
        elif phase == 'test':
            
            # random brightness:
            factor = np.random.choice([1, 1.25, 1.5, 2], p = (0.30, 0.35, 0.25, 0.1))
            factor = 1.0 / factor
            table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype("uint8")
            image = cv2.LUT(image, table)
            
            # random histogram equalizer
            if random.randint(0,1):
                img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
                img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
                image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB) # to RGB
            
            return image
        
        else:
            raise Exception('No valid phase. Training phase takes the argument of "train" or "test".')
            
            
    def _bounding_box_crop(self, image, filename):

        xmax = self.bounding_box['xmax'][filename]
        xmin = self.bounding_box['xmin'][filename]
        ymax = self.bounding_box['ymax'][filename]
        ymin = self.bounding_box['ymin'][filename]
        
        image = image[int(ymin):int(ymax), int(xmin):int(xmax),:]
        
        return image
            
    
    def __len__(self):
        return len(self.datacsv) // self.batch_size
        
        
    def __getitem__(self, index):
        
        # sample batch from dataset
        batch_index = self.indices[index*self.batch_size: (index+1)*self.batch_size]
        batch_fnames = self.datacsv['Filename'].iloc[batch_index].tolist()
        
        # make empty list for input and output
        im = []; label = []
    
        # make imgs
        for idx, fname in enumerate(batch_fnames):
            
            # load image
            impath = os.path.join('/data/volume03/kaggle/train_stage2/image', fname)
            image = cv2.imread(impath)
            
            # random bounding box crop (p=0.8)
            if self.bounding_box:
                if np.random.uniform() < 0.8:
                    image = self._bounding_box_crop(image, fname)
            
            image = self.augmentation(image, self.phase)/255.
            image = cv2.resize(image, (self.image_size, self.image_size))
            
            im.append(image)
            label.append(self.target[fname])
            
        if self.num_ensemble:
            image_input = [np.array(im) for _ in range(self.num_ensemble)]
            return image_input, np.float32(label)
        
        return np.array(im), np.float32(label)

        
    def on_epoch_end(self):
        self.indices = np.arange(len(self.datacsv))
        if self.shuffle:
            np.random.shuffle(self.indices)