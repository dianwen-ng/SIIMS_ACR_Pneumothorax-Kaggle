import pandas as pd
import numpy as np
import os as os
import inspect
import collections
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import keras
from glob import glob
from PIL import Image
import cv2
from os.path import splitext, basename

#############################################################
## Note: There are 37 images in training set not diagnosed ##
##          (we will remove them from training)            ##
#############################################################

# set seed
np.random.seed(609)

# Define train data root
curr_filename = inspect.getfile(inspect.currentframe())
root_dir = os.path.dirname(os.path.abspath(curr_filename))
train_root_dir = os.path.join(root_dir, 'train_stage2')

def load_imgFilename(train_root_dir=train_root_dir, load_positive=False):
    
    curr_filename = inspect.getfile(inspect.currentframe())
    root_dir = os.path.dirname(os.path.abspath(curr_filename))
    # read diagnosed csv data
    csv_path = os.path.join(root_dir, 'siim-acr-pneumothorax-segmentation-2/stage_2_train.csv') 
    train_csv = pd.read_csv(csv_path) # Insert dir path of csv file here (/root/train/.)
    
    # loading only positive images
    if load_positive:
        positive_cases = train_csv.ImageId.values[(train_csv['EncodedPixels'].values != '-1')]
        positive_imnames = collections.Counter(positive_cases)
        imcsv_name = [name + '.png' for name in positive_imnames.keys()]
    
    else:
        imcsv_name = np.unique(train_csv.ImageId.values)
        imcsv_name = [name + '.png' for name in imcsv_name]
    
    unicsv_name = list(dict.fromkeys(imcsv_name))
        
    # Define image path
    assert os.path.isdir(train_root_dir), \
    'There are some errors in the function causing it to stop running. Consider checking the path name of the images'
    
    im_name = [fname for fname in os.listdir(os.path.join(train_root_dir, 'image')) \
               if fname.startswith('1.2.276.0.7230010.3.1.4.8323329') and \
               fname.lower().endswith('.png')]
    unlisted_IMG = list(set(im_name) - set(unicsv_name))
   
    im_fname = list(set(im_name) - set(unlisted_IMG)) 
    
    # Load image with desired resolution; i.e. (1024,1024)
    imgList = [os.path.join(train_root_dir, 'image', i) \
               for i in im_fname]
    maskList = [os.path.join(train_root_dir, 'mask', i) \
                for i in im_fname]
    
    
    return imgList, maskList

# split data list into train & validation
def split_datalist(train_image,
                   train_mask,
                   split_ratio = 0.8):
    
    # Sanity check on image and mask path dir
    assert len(train_image) == len(train_mask), \
    "Error in dimension of train image or train mask. Check the corresponding image, label pairs"
    
    # Set split threshold
    split_threshold = int(split_ratio * len(train_image))
    zip_data = list(zip(train_image,train_mask))
    random.shuffle(zip_data)
    img, mask = zip(*zip_data)
    
    training_img = img[:split_threshold]
    valid_img = img[split_threshold:]
    
    training_mask = mask[:split_threshold]
    valid_mask = mask[split_threshold:]
    
    return training_img, valid_img, training_mask, valid_mask

# load full train data
train_imgList, train_maskList = load_imgFilename(train_root_dir = train_root_dir, load_positive=False)
imgs_train, imgs_valid, label_train, label_valid = split_datalist(train_imgList, train_maskList, split_ratio=0.8)

# load only positive data
pos_img, pos_mask = load_imgFilename(train_root_dir = train_root_dir, load_positive=True)
imtrain_pos, imvalid_pos, labtrain_pos, labvalid_pos = split_datalist(pos_img, pos_mask, split_ratio = 0.8)

# get path for only negative case
neg_img, neg_mask = list(set(train_imgList) - set(pos_img)), list(set(train_maskList) - set(pos_mask))
imtrain_neg, imvalid_neg, labtrain_neg, labvalid_neg = split_datalist(train_image = neg_img, train_mask = neg_mask, split_ratio = 0.95)

stratified_valid_im = imvalid_pos + imvalid_neg
stratified_valid_lab = labvalid_pos + labvalid_neg

# summary of loading data split
# positive samples loaded in data_pipeline: pos_img-> imtrain_pos, imvalid_pos, pos_mask-> labtrain_pos, labvalid_pos
# full training data: train_imgList-> imgs_train, imgs_valid, train_maskList-> label_train, label_valid

#############################################################
####           Data pipline for Keras model              ####
#############################################################    

# Build data generator
# Set params
h, w, batch_size = 512, 512, 32

# load bounding box label:
# these predicted boxes are generated from frcnn by fitting the labels 
# from DR Konya's hand labelled truth posted in forum.

bounding_box1 = pd.read_csv('./boundingbox_csv/siim_pneumothorax_train_bb.csv') 
bounding_box2 = pd.read_csv('./boundingbox_csv/nih_train_bb.csv')
bounding_box3 = pd.read_csv('./boundingbox_csv/siim_pneumothorax_test_bb.csv')
bounding_box4 = pd.read_csv('./boundingbox_csv/stage2bb.csv')
bbox_total = pd.concat([bounding_box1, bounding_box2, bounding_box3, bounding_box4])

def make_crop_box(image, label, image_path, crop_shape=(749,949)):
    if np.random.uniform() < 0.5:
        filename = basename(image_path)
        box_coord = bbox_total.loc[bbox_total.filename == filename]
        img = image[int(box_coord.ymin):int(box_coord.ymax), int(box_coord.xmin):int(box_coord.xmax)]
        lab = label[int(box_coord.ymin):int(box_coord.ymax), int(box_coord.xmin):int(box_coord.xmax)]
        
    elif np.random.uniform() < 0.15:
        img,lab = random_crop(image, label, crop_shape)

    else:
        img, lab = image, label
        
    return img, lab

def random_crop(image, label, crop_shape):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
        
    if (crop_shape[0] < image.shape[1]) and (crop_shape[1] < image.shape[0]):
        x = random.randrange(image.shape[1]-crop_shape[0])
        y = random.randrange(image.shape[0]-crop_shape[1])
        
        img = image[y:y+crop_shape[1], x:x+crop_shape[0]]
        lab = label[y:y+crop_shape[1], x:x+crop_shape[0]]
        
        return img, lab


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, im_path, mask_path, 
                 augmentations=None, batch_size=30, img_size=512,
                 n_channels=3, shuffle=True, by_ratio=0.0,
                 num_ensemble=False, box_cropping = True):

        'Initialization'
        self.batch_size = batch_size
        self.im_paths = im_path
        self.mask_path = mask_path

        self.img_size = img_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augmentations
        self.ensemble = num_ensemble
        self.by_ratio = by_ratio
        self.box_cropping = box_cropping
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of iterations(steps) per epoch'
        if np.ceil(self.by_ratio) == True:
            length = int(np.floor(len(self.im_paths[1]) / (self.batch_size)))
            return length
        else:
            length = int(np.ceil(len(self.im_paths) / self.batch_size))
            return length

    def __getitem__(self, index):
        'Generate one batch of data'
        if np.ceil(self.by_ratio) == True:
            num_positive = int(self.batch_size*self.by_ratio)
            num_negative = self.batch_size - num_positive
            # since positive case is lesser, we let positive case to oversample by cycling samples
            pos_index = int(index % int(np.floor(len(self.im_paths[0]) / (self.batch_size*self.by_ratio))))
            
            indexes_positive = self.indexes_pos[pos_index*num_positive:
                                            min((pos_index+1)*num_positive, len(self.im_paths[0]))]

            indexes_negative = self.indexes_neg[index*num_negative:
                                            min((index+1)*num_negative, len(self.im_paths[1]))]

            # Find list of IDs
            list_IDs_im = [self.im_paths[0][k] for k in indexes_positive] + [self.im_paths[1][k] for k in indexes_negative]
            list_IDs_lab = [self.mask_path[0][k] for k in indexes_positive] + [self.mask_path[1][k] for k in indexes_negative]
            list_IDs_im = zip(list_IDs_im,list_IDs_lab)

        else:
            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:min((index+1)*self.batch_size,len(self.im_paths))]
            # Find list of IDs
            list_IDs_im = zip([self.im_paths[k] for k in indexes],[self.mask_path[k] for k in indexes])

        
        # Generate data
        X, y = self.data_generation(list_IDs_im)

        if self.augment is None:
            inputX = X/255
            
            if self.ensemble:
                inputX = [inputX for _ in range(self.ensemble + 1)]
            mask = y
            
        else:            
            im,mask = [],[]   
            for x,y in zip(X,y):
                augmented = self.augment(image=x, mask=y)
                im.append(augmented['image'])
                mask.append(augmented['mask'])
                
            if self.ensemble:
                inputX = np.array(im)/255
                inputX = [inputX for _ in range(self.ensemble + 1)]
                
            else:
                inputX = np.array(im)/255
                
        return inputX, np.int8(np.array(mask)>0)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if np.ceil(self.by_ratio) == True:
            self.indexes_pos = np.arange(len(self.im_paths[0]))
            self.indexes_neg = np.arange(len(self.im_paths[1]))
            if self.shuffle == True:
                np.random.shuffle(self.indexes_pos)
                np.random.shuffle(self.indexes_neg)
        
        else:
            self.indexes = np.arange(len(self.im_paths))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

    def data_generation(self, list_IDs_im):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size,self.img_size,self.img_size, self.n_channels))
        y = np.empty((self.batch_size,self.img_size,self.img_size, 1))

        # Generate data
        for i, (im_path,mask_path) in enumerate(list_IDs_im):
            
            im = np.array(Image.open(im_path))            
            mask = np.array(Image.open(mask_path))
            mask = mask.astype('uint8')
            
            # random cropping here
            if self.box_cropping:
                im, mask = make_crop_box(im, mask, im_path)
            
            if len(im.shape)==2:
                im = np.repeat(im[...,None],3,2)

            # Resize sample
            X[i,] = cv2.resize(im, (self.img_size,self.img_size))

            # Store class
            y[i,] = cv2.resize(mask,(self.img_size,self.img_size))[..., np.newaxis]
            y[y>0] = 255

        return np.uint8(X),np.uint8(y)


def train_with_generator(model, epochs, train_generator, valid_generator,
                         callbacks, workers = 12, mp = False):
    # Define num of iterations
    steps = len(train_generator)
    # Training model
    training = model.fit_generator(generator=train_generator,
                                   steps_per_epoch=steps, 
                                   epochs = epochs, verbose=1, 
                                   callbacks = callbacks,
                                   validation_data=valid_generator,
                                   validation_steps=len(valid_generator),
                                   max_queue_size=12,
                                   workers=workers, use_multiprocessing=mp)
    
    return training   


