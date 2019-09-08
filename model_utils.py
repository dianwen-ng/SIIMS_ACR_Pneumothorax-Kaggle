import tensorflow as tf
from tensorflow.python.layers import base
from model_metrics import *
import os
import inspect 
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, CSVLogger, EarlyStopping
from keras.optimizers import Adam
import keras


# Define train data root
curr_filename = inspect.getfile(inspect.currentframe())
root_dir = os.path.dirname(os.path.abspath(curr_filename))

# callbacks params and configs
class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr

    def get_callbacks(self, model_prefix='Model'):
        
        # Save checkpoint based on kaggle_iou and jaccard
        path = os.path.join(root_dir, 'checkpoints_stack_seg')
        
        filepath = os.path.join(path,'improved_iou_{epoch:02d}_{val_my_iou_metric:.2f}.hdf5')
        checkpointer = ModelCheckpoint(filepath = filepath, verbose=1, save_best_only=True,
                                       save_weights_only=True, monitor = 'val_my_iou_metric', mode = 'max')
        
        logger = CSVLogger(os.path.join(path,'training_log.csv'))

        # early stoppage
        stop_train = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=50, verbose=1, mode = 'auto')
        # reduce learning rate scheduling
        schedule_lr = keras.callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule)
        # schedule_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, mode = 'min', patience=1, min_lr=1e-8, cooldown=1)

        # design callbacks
        callback_list = [schedule_lr, stop_train, checkpointer, logger]
        
        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M)) # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        lr= float(self.alpha_zero / 2 * cos_out)
        
        # reduce learning rate as training progress 
        if not t%25:
            self.alpha_zero *= 0.5
            
        return lr
        

# compile models
def compile_model(model, num_classes, metrics, loss, lr):
    from model_metrics import dice_coeff, jaccard_index, class_jaccard_index, bce_dice_loss, my_iou_metric, focal_loss
    from model_metrics import pixelwise_precision, pixelwise_sensitivity, pixelwise_specificity, pixelwise_recall
    
    if isinstance(loss, str):
        if loss in {'ce', 'crossentropy'}:
            if num_classes == 1:
                loss = keras.losses.binary_crossentropy
            else:
                loss = keras.losses.categorical_crossentropy
        elif loss in {'bce_dice', 'binary_ce_dice'}:
            loss = bce_dice_loss
        elif loss == 'weighted_bce_dice':
            loss = weighted_cross_entropy_dice(2)
        elif loss == 'weighted_bce':
            loss = weighted_ce(2)
        elif loss == 'mixed':
            loss = Mixedloss(alpha=0.25, gamma=2)
        elif loss == 'focal':
            loss = focal_loss(alpha=0.5, gamma=2)
        else:
            raise ValueError('unknown loss %s' % loss)

    if isinstance(metrics, str):
        metrics = [metrics, ]

    for i, metric in enumerate(metrics):
        if not isinstance(metric, str):
            continue
        elif metric == 'acc':
            metrics[i] = keras.metrics.binary_accuracy if num_classes == 1 \
            else keras.metrics.categorical_accuracy
        elif metric == 'jaccard_index':
            metrics[i] = jaccard_index(num_classes)
        elif metric == 'jaccard_index0':
            metrics[i] = class_jaccard_index(0)
        elif metric == 'jaccard_index1':
            metrics[i] = class_jaccard_index(1)
        elif metric == 'jaccard_index2':
            metrics[i] = class_jaccard_index(2)
        elif metric == 'jaccard_index3':
            metrics[i] = class_jaccard_index(3)
        elif metric == 'jaccard_index4':
            metrics[i] = class_jaccard_index(4)
        elif metric == 'jaccard_index5':
            metrics[i] = class_jaccard_index(5)
        elif metric == 'dice_score':
            metrics[i] = dice_coeff(num_classes)
        elif metric == 'pixelwise_precision':
            metrics[i] = pixelwise_precision(num_classes)
        elif metric == 'pixelwise_sensitivity':
            metrics[i] = pixelwise_sensitivity(num_classes)
        elif metric == 'pixelwise_specificity':
            metrics[i] = pixelwise_specificity(num_classes)
        elif metric == 'pixelwise_recall':
            metrics[i] = pixelwise_recall(num_classes)
        elif metric == 'iou':
            metrics[i] = my_iou_metric
        
 
        else:
            raise ValueError('metric %s not recognized' % metric)


    model.compile(Adam(lr),
                  loss=loss,
                  metrics=metrics)
    
    
    
class SWA(keras.callbacks.Callback):
    
    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch 
    
    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.swa_epoch))
        
    def on_epoch_end(self, epoch, logs=None):
        
        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()
            
        elif epoch > self.swa_epoch:    
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (self.swa_weights[i] * 
                    (epoch - self.swa_epoch) + self.model.get_weights()[i])/((epoch - self.swa_epoch)  + 1)  

        else:
            pass
        
    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save_weights(self.filepath)
        print('Final stochastic averaged weights saved to file.')
        
        
        
    