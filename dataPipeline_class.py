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
from os.path import basename

datapath=pd.read_csv('class_labels.csv')

curr_filename = inspect.getfile(inspect.currentframe())
root_dir = os.path.dirname(os.path.abspath(curr_filename))
train_root_dir = os.path.join(root_dir, 'train/images/1024/dicom/')

dataImg_path = [os.path.join(train_root_dir,img) for img in datapath.image]
label = datapath.labels
label = keras.utils.to_categorical(label)

# split data list into train & validation
def split_datalist(train_image = dataImg_path,
                   train_label = label,
                   split_ratio = 0.8):
    
    # Sanity check on image and mask path dir
    assert len(train_image) == len(train_label), \
    "Error in dimension of train image or train mask. Check the corresponding image, label pairs"
    # Set split threshold
    split_threshold = int(split_ratio * len(train_image))
    training_img = train_image[:split_threshold]
    valid_img = train_image[split_threshold:]
    
    training_labels = train_label[:split_threshold]
    valid_labels = train_label[split_threshold:]
    
    return training_img, valid_img, training_labels, valid_labels
    
imgs_train, imgs_valid, label_train, label_valid = split_datalist(split_ratio=0.8)


class ClassificationGenerator(keras.utils.Sequence):
    
    def __init__(self, image_path = dataImg_path, label_path = label,
                 n_classes=2, batch_size=16, resize_shape=(512,512), elastic_transform = True,
                 seed = 610, crop_shape=False, horizontal_flip=True, blur = False,
                 vertical_flip=False, brightness=True, rotation=2.0, zoom=False, do_ahisteq = True,
                 bbox = True, ensemble=False):
        
        # Set configs
        self.image_path_env = image_path
        self.label_path_env = label_path
        self.blur = blur
        self.histeq = do_ahisteq
        self.image_path_list = tuple(list(self.image_path_env).copy())
        self.label_path_list = tuple(list(self.label_path_env).copy())

        np.random.seed(seed)
        # Shuffle data 
        x = np.random.permutation(len(self.image_path_list))   
        self.image_path_list = [self.image_path_list[j] for j in x]
        self.label_path_list = [self.label_path_list[j] for j in x]
        
        # Augmentation configs
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.brightness = brightness
        self.rotation = rotation
        self.zoom = zoom
        self.elastic_transform = elastic_transform
        self.bbox = bbox
        self.ensemble = ensemble

        # Preallocate memory
        if isinstance(self.resize_shape[0], int) and self.resize_shape[0] == self.resize_shape[1]:
            self.X = np.zeros((batch_size, resize_shape[1], resize_shape[0], 3), dtype='float32')
            self.Y = np.zeros((batch_size, 2), dtype='float32')

        else:
            raise Exception('No image dimensions specified!')

        
    def __len__(self):
        return len(self.image_path_list) // self.batch_size

    
    def __getitem__(self, i):
    
        # make imgs
        for n, (image_path, label_path) in enumerate(zip(self.image_path_list[i*self.batch_size:(i+1)*self.batch_size], self.label_path_list[i*self.batch_size:(i+1)*self.batch_size])):
            
            # Read image, mask
            image = cv2.imread(image_path, 1)
            
            if self.bbox:
                image = _make_crop_box(image, image_path)
            
            if self.blur and random.randint(0,1):
                image = cv2.GaussianBlur(image, (self.blur, self.blur), 0)

            if self.crop_shape and random.randint(0,1):
                image, label = _random_crop(image, label, self.crop_shape)
                
            if self.resize_shape:
                # Add gaussian noise (next time)
                image = cv2.resize(image, self.resize_shape) 
               
            # Do augmentation                
            if self.horizontal_flip and random.randint(0,1):
                image = cv2.flip(image, 1)
                
            if self.vertical_flip and random.randint(0,1):
                image = cv2.flip(image, 0)
            
            if self.brightness:
                factor = np.random.choice([1,1.25,1.5,2], p = (0.30,0.35,0.25,0.1))
                factor = 1.0/factor
                table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype("uint8")
                image = cv2.LUT(image, table)
            
            if self.rotation:
                angle = random.gauss(mu=0.0, sigma=self.rotation)
            else:
                angle = 0.0
            
            if self.zoom:
                scale = random.gauss(mu=1.0, sigma=self.zoom)
            else:
                scale = 1.0
            
            if self.rotation and self.zoom:
                M = cv2.getRotationMatrix2D((image.shape[1]//2, image.shape[0]//2), angle, scale)
                image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

            if self.histeq and random.randint(0,1): # and convert to RGB
                img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
                img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
                image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB) # to RGB
                
            elif random.randint(0,1):
                image = _contrast(image, 8)
                
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR to RGB
                
            if self.elastic_transform and (random.randint(0,1)*random.randint(0,1)):
                image = _elastic_transform(image)

            # Make image
            self.X[n] = image/255. 
            
            # Make label
            self.Y[n] = label_path
        
        if self.ensemble:
            inputX = [self.X for _ in range(self.ensemble)]
        else:
            inputX = self.X
            
        
        return inputX, self.Y
        
    def on_epoch_end(self):
        # Shuffle dataset for next epoch
        c = list(zip(self.image_path_env, self.label_path_env))
        random.shuffle(c)
        self.image_path_env, self.label_path_env = zip(*c)
        
        
        
    
def _random_crop(image, label, crop_shape):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
        
    if (crop_shape[0] < image.shape[1]) and (crop_shape[1] < image.shape[0]):
        x = random.randrange(image.shape[1]-crop_shape[0])
        y = random.randrange(image.shape[0]-crop_shape[1])
        
        return image[y:y+crop_shape[1], x:x+crop_shape[0], :], label[y:y+crop_shape[1], x:x+crop_shape[0]]
    else:
        image = cv2.resize(image, crop_shape)
        label = cv2.resize(label, crop_shape, interpolation = cv2.INTER_NEAREST)
        return image, label
    
    
# Function to distort image
def _elastic_transform(image, alpha=120, sigma=100*0.16, alpha_affine=100*0.04, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    
    pts1 = np.float32([center_square + square_size,
                       [center_square[0]+square_size, center_square[1]-square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    
    return image

def _contrast(image, value=16):
    f = 131*(value + 127)/(127*(131-value))
    alpha_c = f
    gamma_c = 127*(1-f)
    image = cv2.addWeighted(image, alpha_c, image, 0 ,gamma_c)
    return image
  
bounding_box = pd.read_csv('./boundingbox_csv/siim_pneumothorax_train_bb.csv') 
bounding_box2 = pd.read_csv('./nih_train_bb.csv')
bbox_total = pd.concat([bounding_box, bounding_box2])

def _make_crop_box(image, image_path):
    if np.random.uniform() < 0.8:
        filename = basename(image_path)
        box_coord = bbox_total.loc[bbox_total.filename == filename]
        img = image[int(box_coord.ymin):int(box_coord.ymax), int(box_coord.xmin):int(box_coord.xmax)]
    
    else:
        img = image
        
    return img
        
        

def create_generators(image_path=imgs_train, label_path=label_train, 
                      n_classes = 1, batch_size = 16, 
                      resize_shape=(512,512), elastic_transform = True, seed=610,
                      crop_shape = False, horizontal_flip = True,
                      blur = False, vertical_flip = False,
                      brightness=0.2, rotation=2.0, bbox = True,
                      zoom=False, do_ahisteq = True, ensemble=False):
       
    generator = ClassificationGenerator(image_path=image_path, label_path=label_path,
                                        n_classes = n_classes, batch_size=batch_size,
                                        resize_shape=resize_shape, 
                                        elastic_transform=elastic_transform, seed=seed, 
                                        crop_shape=crop_shape, horizontal_flip=horizontal_flip,
                                        blur = blur, vertical_flip=vertical_flip,
                                        brightness=brightness, rotation=rotation,
                                        zoom=zoom, do_ahisteq = do_ahisteq, bbox=bbox, ensemble=ensemble)
    
    
    return generator


def train_with_generator(model, epochs, train_generator, valid_generator, 
                         callbacks, class_weights=None, workers = 12, mp = False):
    # Define num of iterations
    steps = len(train_generator)
    # Training model
    training = model.fit_generator(generator=train_generator,
                                   steps_per_epoch=steps, 
                                   epochs = epochs, verbose=1, 
                                   callbacks = callbacks,
                                   validation_data=valid_generator,
                                   validation_steps=len(valid_generator),
                                   class_weight=class_weights,
                                   max_queue_size=12,
                                   workers=workers, use_multiprocessing=mp)
    
    return training   