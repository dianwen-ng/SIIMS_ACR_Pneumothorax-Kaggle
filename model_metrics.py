import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy
import numpy as np
import keras
import matplotlib.pyplot as plt


def pixelwise_precision(num_classes=1):
    def binary_pixelwise_precision(y_true, y_pred):
        true_pos = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        total_pos = K.sum(K.abs(y_pred), axis=[1, 2, 3])
        return true_pos / K.clip(total_pos, K.epsilon(), None)

    def categorical_pixelwise_precision(y_true, y_pred):
        true_pos = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
        total_pos = K.sum(K.abs(y_pred), axis=[1, 2])
        return true_pos / K.clip(total_pos, K.epsilon(), None)

    if num_classes == 1:
        return binary_pixelwise_precision
    else:
        return categorical_pixelwise_precision

    
def pixelwise_recall(num_classes=1):
    return pixelwise_sensitivity(num_classes)


def pixelwise_sensitivity(num_classes=1):
    def binary_pixelwise_sensitivity(y_true, y_pred):
        """
        true positive rate, probability of detection

        sensitivity = # of true positives / (# of true positives + # of false negatives)

        Reference: https://en.wikipedia.org/wiki/Sensitivity_and_specificity
        :param y_true:
        :param y_pred:
        :return:
        """
        # indices = tf.where(K.greater_equal(y_true, 0.5))
        # y_pred = tf.gather_nd(y_pred, indices)

        y_true = K.round(y_true)
        true_pos = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        total_pos = K.sum(K.abs(y_true), axis=[1, 2, 3])
        return true_pos / K.clip(total_pos, K.epsilon(), None)

    def categorical_pixelwise_sensitivity(y_true, y_pred):
        true_pos = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
        total_pos = K.sum(K.abs(y_true), axis=[1, 2])
        return K.mean(true_pos / K.clip(total_pos, K.epsilon(), None), axis=-1)

    if num_classes == 1:
        return binary_pixelwise_sensitivity
    else:
        return categorical_pixelwise_sensitivity


    
def pixelwise_specificity(num_classes=1):
    """
    true negative rate
    the proportion of negatives that are correctly identified as such

    specificity = # of true negatives / (# of true negatives + # of false positives)

    :param y_true:  ground truth
    :param y_pred: prediction
    :return:
    """
    def binary_pixelwise_specificity(y_true, y_pred):
        true_neg = K.sum(K.abs((1. - y_true) * (1. - y_pred)), axis=[1, 2, 3])
        total_neg = K.sum(K.abs(1. - y_true), axis=[1, 2, 3])
        return true_neg / K.clip(total_neg, K.epsilon(), None)

    def categorical_pixelwise_specificity(y_true, y_pred):
        y_true, y_pred = y_true[..., 1:], y_pred[..., 1:]
        true_neg = K.sum(K.abs((1. - y_true) * (1. - y_pred)), axis=[1, 2])
        total_neg = K.sum(K.abs(1. - y_true), axis=[1, 2])
        return true_neg / K.clip(total_neg, K.epsilon(), None)
    
    if num_classes == 1:
        return binary_pixelwise_specificity
    else:
        return categorical_pixelwise_specificity


    
def dice_coeff(num_classes=1):
    def binary_dice_coeff(y_true, y_pred):
        """
                DSC = (2 * |X & Y|)/ (|X|+ |Y|)
                    = 2 * sum(|A*B|)/(sum(|A|)+sum(|B|))
                    
        :param y_true: ground truth
        :param y_pred: prediction
        :return:
        """
        y_true_f = K.flatten(y_true)
        y_pred = K.cast(y_pred, 'float32')
        y_pred_f = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
        intersection = y_true_f * y_pred_f
        score = 2. * K.sum(intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
        return score

    def categorical_dice_coeff(y_true, y_pred):

        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
        union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1, 2])
        dice = 2 * intersection / K.clip(union, K.epsilon(), None)
        return K.mean(dice, axis=-1)

    if num_classes == 1:
        return binary_dice_coeff
    else:
        return categorical_dice_coeff
    
    
    
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def bce_logdice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))

def weighted_ce(beta):
    # beta-weight for positive case
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        return K.log(y_pred / (1 - y_pred))
    
    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)
        return tf.reduce_mean(loss)
    
    return loss

def weighted_cross_entropy_dice(beta):
    # beta-weight for positive case
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        return K.log(y_pred / (1 - y_pred))
    
    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)
        return tf.reduce_mean(loss)
    
    def weighted_bce_dice(y_true, y_pred):
        return loss(y_true, y_pred)- K.log(1. - dice_loss(y_true, y_pred))
    
    return weighted_bce_dice


def class_jaccard_index(idx):
    def jaccard_index(y_true, y_pred):
        y_true, y_pred = y_true[..., idx], y_pred[..., idx]
        y_true = K.round(y_true)
        y_pred = K.round(y_pred)
        # Adding all three axis to average across images before dividing
        # See https://forum.isic-archive.com/t/task-2-evaluation-and-superpixel-generation/417/2
        intersection = K.sum(K.abs(y_true * y_pred), axis=[0, 1, 2])
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[0, 1, 2])
        jac = intersection / K.clip(sum_ - intersection, K.epsilon(), None)
        return jac
    return jaccard_index


def jaccard_index(num_classes):
    """
    Jaccard index for semantic segmentation, also known as the intersection-over-union.

        This loss is useful when you have unbalanced numbers of pixels within an image
        because it gives all classes equal weight. However, it is not the defacto
        standard for image segmentation.

        For example, assume you are trying to predict if each pixel is cat, dog, or background.
        You have 80% background pixels, 10% dog, and 10% cat. If the model predicts 100% background
        should it be be 80% right (as with categorical cross entropy) or 30% (with this loss)?

        The loss has been modified to have a smooth gradient as it converges on zero.
        This has been shifted so it converges on 0 and is smoothed to avoid exploding
        or disappearing gradient.

        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

        # References

        Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
        What is a good evaluation measure for semantic segmentation?.
        IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.

        https://en.wikipedia.org/wiki/Jaccard_index

        """

    def binary_jaccard_index(y_true, y_pred):
        y_true = K.round(y_true)
        y_pred = K.round(y_pred)
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=[1, 2, 3])
        iou = intersection / K.clip(union - intersection, K.epsilon(), None)
        return iou

    def categorical_jaccard_index(y_true, y_pred):
        y_true = K.round(y_true)
        y_pred = K.round(y_pred)
        intersection = K.abs(y_true * y_pred)
        union = K.abs(y_true) + K.abs(y_pred)

        intersection = K.sum(intersection, axis=[0, 1, 2])
        union = K.sum(union, axis=[0, 1, 2])

        iou = intersection / K.clip(union - intersection, K.epsilon(), None)
        # iou = K.mean(iou, axis=-1)
        return iou

    if num_classes == 1:
        return binary_jaccard_index
    else:
        return categorical_jaccard_index
    
    
    
def focal_loss(alpha=1, gamma=2):
    
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
        return (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 
    
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        logits = tf.log(y_pred / (1 - y_pred))
        
        loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
        return tf.reduce_mean(loss)
    
    return loss
 


def Mixedloss(alpha=1, gamma=2):   
    
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
        return (tf.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b 
    
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        logits = tf.log(y_pred / (1 - y_pred))
        
        loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
        return tf.reduce_mean(loss)
    
    def Mixedlosses(y_true, y_pred):
        return loss(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))
    
    return Mixedlosses


    
# https://www.kaggle.com/cpmpml/fast-iou-metric-in-numpy-and-tensorflow
def get_iou_vector(A, B):
    # Numpy version    
    batch_size = A.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        true = np.sum(t)
        pred = np.sum(p)
        
        # deal with empty mask first
        if true == 0:
            metric += (pred == 0)
            continue
        
        # non empty mask case.  Union is never empty 
        # hence it is safe to divide by its number of pixels
        intersection = np.sum(t * p)
        union = true + pred - intersection
        iou = intersection / union
        
        # iou metrric is a stepwise approximation of the real iou over 0.5
        iou = np.floor(max(0, (iou - 0.45)*20)) / 10
        
        metric += iou
        
    # teake the average over all images in batch
    metric /= batch_size
    return metric



def my_iou_metric(label, pred):
    # Tensorflow version
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)


from sklearn.metrics import confusion_matrix
# plot confusion_matrix
def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    print('sensitivity: ', cm[1,1]/sum(cm[1,:]))
    print('specificity: ', cm[0,0]/sum(cm[0,:]))

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax, cm