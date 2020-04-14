import os as os
import random
import numpy as np
import cv2
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

class DataGenerator(keras.utils.Sequence):

    def __init__(self, data_csv,
                 image_size=512,
                 batch_size=32,
                 augmentation=None, # wrapper to perform augmentations for train/testing
                 bounding_box=None,
                 shuffle=True,
                 ratio=0.7,
                 num_ensemble=None,
                 seed=612):
        
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.datacsv = data_csv
        self.positive_set = data_csv[data_csv['Label']==1]
        self.negative_set = data_csv[data_csv['Label']==0]
        self.image_size = image_size
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.ratio = ratio
        self.num_ensemble = num_ensemble
        
        # find max length positive, negative
        if not (self.ratio == None or self.ratio == 0):
            self.max_positive_index = len(self.positive_set) // int(self.batch_size * self.ratio)
            self.max_negative_index = len(self.negative_set) // int(self.batch_size * (1-self.ratio))
        
        # convert bounding box csv to hash table if provided
        if bounding_box is not None:
            self.bounding_box = bounding_box.set_index('Filename').to_dict()
            
        else:
            self.bounding_box = bounding_box

        self.on_epoch_end()
        
        
        
    def __len__(self):
        
        if not (self.ratio == None or self.ratio == 0):
            return len(self.negative_set) // int((1-self.ratio) * self.batch_size)
        
        else:
            return len(self.datacsv) // self.batch_size -1

        
        
    def __getitem__(self, index):
        ## processing for over-sampling
        if not (self.ratio == None or self.ratio == 0):
            
            pos_start_index = (index % self.max_positive_index) * int(self.ratio*self.batch_size)
            pos_end_index = ((index+1) % self.max_positive_index) * int(self.ratio*self.batch_size)
            pos_index = list(self.positive_indices[pos_start_index : pos_end_index])
            
            neg_start_index = (index % self.max_negative_index) * int((1-self.ratio)*self.batch_size)
            neg_end_index = ((index+1) % self.max_negative_index) * int((1-self.ratio)*self.batch_size)
            neg_index = list(self.negative_indices[neg_start_index : neg_end_index])
            
            # make fname list
            batch_fnames = self.positive_set['Filename'].iloc[pos_index].tolist() + \
                           self.negative_set['Filename'].iloc[neg_index].tolist()
        
        else: 
            ## processing for non-sampling
            batch_index = self.indices[index*self.batch_size : (index+1)*self.batch_size]
            batch_fnames = self.datacsv['Filename'].iloc[batch_index].tolist()

        ## generate data
        images, masks = self.data_generation(batch_fnames)

        if self.augmentation:
            imgs, labs = [], []
            for im, lab in zip(images, masks):
                augmented = self.augmentation(image=im, mask=lab)
                imgs.append(augmented['image'])
                labs.append(augmented['mask'])
            
            imgs, labs = np.array(imgs)/255, np.float32(labs)
        else:
            # standardization to range (0, 1)
            imgs = images/255
            labs = np.float32(masks)
            
        ## make ensembling output    
        if self.num_ensemble:
            imgs = [imgs for _ in range(self.num_ensemble + 1)]
            
        return imgs, labs
            
        
        
    def on_epoch_end(self):
        
        ## Shuffle dataset for next epoch
        if self.ratio:
            self.positive_indices = np.arange(len(self.positive_set))
            self.negative_indices = np.arange(len(self.negative_set))
            
            if self.shuffle:
                np.random.shuffle(self.positive_indices)
                np.random.shuffle(self.negative_indices)

        else:
            self.indices = np.arange(len(self.datacsv))
            
            if self.shuffle:
                np.random.shuffle(self.indices)
                
                
                
    def data_generation(self, fnames_list):
        
        # Initialization
        X = np.empty((self.batch_size, self.image_size, self.image_size, 3))
        y = np.empty((self.batch_size, self.image_size, self.image_size, 1))

        # Generate data
        for i, fname in enumerate(fnames_list):
            
            imPath = os.path.join('train_stage2/image', fname)
            maskPath = os.path.join('train_stage2/mask', fname)
            
            im = cv2.imread(imPath)
            mask = cv2.imread(maskPath).astype('uint8')
            
            # random cropping 
            if self.bounding_box:
                im, mask = self._random_crop(im, mask, fname)
            
            else:
                # squeeze mask into 1 channel
                mask = mask[:, :, 0]
            
            # resize image, mask
            X[i,] = cv2.resize(im, (self.image_size, self.image_size))
            y[i,] = cv2.resize(mask, (self.image_size, self.image_size))[..., np.newaxis]
            
            # make binary target
            y[y>0] = 255

        return np.uint8(X), np.uint8(y)
    
    
    
    def _random_crop(self, image, mask, filename):
        # def bounding box crop wrapper
        def bounding_box_crop(image, mask, filename):
            xmax = self.bounding_box['xmax'][filename]
            xmin = self.bounding_box['xmin'][filename]
            ymax = self.bounding_box['ymax'][filename]
            ymin = self.bounding_box['ymin'][filename]

            image = image[int(ymin):int(ymax), int(xmin):int(xmax), :]
            mask = mask[int(ymin):int(ymax), int(xmin):int(xmax), 0]

            return image, mask
        
        if random.randint(0,1):
            return bounding_box_crop(image, mask, filename)
        
        # super zoom for p=0.15
        elif np.random.uniform() < 0.15:
            x = random.randrange(image.shape[1] - 749)
            y = random.randrange(image.shape[0] - 949)
            
            image = image[y:y+949, x:x+749, :]
            mask = mask[y:y+949, x:x+749, 0]
            
            return image, mask
        
        return image, mask[:, :, 0]
            


