from keras.applications.xception import Xception
from efficientnet import EfficientNetB4
from keras.layers import Dense, Input
from keras.models import Model
from keras.layers import Conv2D, DepthwiseConv2D, SeparableConv2D
from keras.layers import BatchNormalization, Dropout, Add
from keras.layers import LeakyReLU, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import multi_gpu_model


def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = Add()([x, blockInput])
    return x

###############################
##*     Efficient model     *##  
###############################

def EfficientNet_model(input_shape=(None, None, 3),
                       num_classes=1, 
                       weight = 'imagenet',
                       dropout_rate=0.5):

    backbone = EfficientNetB4(weights=weight,
                              include_top=False,
                              input_shape=input_shape)
    
    img_input = backbone.input

    x = backbone.layers[342].output # consider 257, 154
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(dropout_rate)(x)
    
    x = Conv2D(960, (3, 3), activation=None, padding="same")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Conv2D(960, (3, 3), activation=None, padding="same")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    
    if num_classes == 1:
        predictions=Dense(num_classes+1, activation='sigmoid', name='predictions')(x)
    else:
        predictions=Dense(num_classes+1, activation='softmax', name='predictions')(x)
        
    model = Model(inputs=img_input, outputs=predictions)
    
    if weight is not 'imagenet' and not None:
        print('loading model weights from pretrained...')
        model.load_weights(weight)
        
    return model


################################
##*      Xception model      *##
################################

def xception_model(input_shape=(None, None, 3),
                   num_classes = 1, 
                   weight='imagenet'):
    
    img_input = Input(shape=input_shape)
    backbone = Xception(weights=weight,
                        include_top=False,
                        input_tensor=img_input,
                        input_shape=input_shape,
                        pooling='avg')
    
    x = backbone.layers[103].output # 103
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.5)(x)
    
    x = Conv2D(960, (3, 3), activation=None, padding="same")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Conv2D(960, (3, 3), activation=None, padding="same")(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.7)(x)
    
    if num_classes == 1:
        predictions=Dense(num_classes+1, activation='softmax', name='predictions')(x)
    else:
        predictions=Dense(num_classes+1, activation='softmax', name='predictions')(x)
        
    model = Model(inputs=img_input, outputs=predictions)
    
    if weight is not 'imagenet' and not None:
        print('loading model weights from pretrained...')
        model.load_weights(weight)
    
    return model


## Enable multi GPU training 
class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
           serial-model holds references to the weights in the multi-gpu model.
           '''
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)
        else:
            #return Model.__getattribute__(self, attrname)
            return super(ModelMGPU, self).__getattribute__(attrname)
        
