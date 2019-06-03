import cv2 as cv
import tensorflow as tf
import matplotlib.pyplot as plt

def load_image(path, mask=False, shape=(256,256)):
    """

    This method returns an image and make the corresponding preprocessing before, 
    the image is fed to the model

    params:
    path : String with the path to the image
    mask : Bool, True if the image current image is a mask, False otherwhise

    returns:
    img : numpy array corresponding to the  preprocessed image

    """

    if mask:
        img = cv.imread(path, 0)
        img = cv.resize(img, shape, interpolation = cv.INTER_NEAREST)
    else:
        img = cv.imread(path) / 255.0
        img = cv.resize(img, shape, interpolation = cv.INTER_AREA)
    return img
    
    
def replace_ext(path):
    """
    
    Replace the extension of a file, png (mask extension) to jpg (images extension)
    Masks were saved in png format, to avoid loosing information about the classes
    i.e jpg mixed some classes together when compressing

    params:
    path : String with the path to the image

    """
    return path[:-4] + '.jpg'

def draw_results(history):
    """
    
    Draw results given by a model after training
    params:
    history : keras history
    

    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
def get_weights(data):
    """
    Naive approach weighted loss function, pixel by pixel, waay to slow! 
    Never doing this again, was fool but we learnt!

    """
    num_img , width_height = data.shape
    constant = 1 / 46
    weights = np.zeros((num_img , width_height)) + ( constant / 47)
    #print(weights.shape)
    rows, cols = np.where(data != 0)
    for row in rows:
        for col in cols: 
            weights[row, col] = constant
            
    return weights
    
def myLoss(onehot_labels, logits):
    """

    Weighted loss function, after computing softmax_cross_entropy,  weights are multiplied
    across channels, better that the naive approach but only background is being punished

    params:
    onehot_labels: ground truth
    logits: last layer of the model, without softmax, because it is calculated here

    """
    
    constant = 1 / 47
    _weigths = [constant] * 47 
    _weigths[0] =  constant / 47
    
    class_weights = tf.constant(_weigths)
    
    # deduce weights for batch samples based on their true label
    weights = tf.reduce_sum(class_weights * onehot_labels, axis=1)
    
    # compute your (unweighted) softmax cross entropy loss
    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels= onehot_labels, logits = [logits])
    
    # apply the weights, relying on broadcasting of the multiplication
    weighted_losses = unweighted_losses * weights
    
    # reduce the result to get your final loss
    loss = tf.reduce_mean(weighted_losses)

def generalized_dice_loss(labels, logits):

    """
    Dice loss, didn't work good enough, it was okay when training one example and every class
    in the example was considered as one, i.e in binary segmentation problems dice has better
    performance

    labels: ground truth
    logits: last layer of the model, with softmax

    """


    smooth = 1e-17
    shape = tf.TensorShape(logits.shape).as_list()
    depth = int(shape[-1])
    #labels = tf.one_hot(labels, depth, dtype=tf.float32)
    #logits = tf.nn.softmax(logits)
    weights = 1.0 / (tf.reduce_sum(labels, axis=[0, 1, 2])**2)
    print(weights.shape)
    numerator = tf.reduce_sum(labels * logits, axis=[0, 1, 2])
    numerator = tf.reduce_sum(weights * numerator)

    denominator = tf.reduce_sum(labels + logits, axis=[0, 1, 2])
    denominator = tf.reduce_sum(weights * denominator)

    loss = 1.0 - 2.0*(numerator + smooth)/(denominator + smooth)
    return loss

def i_over_u(y_true, y_pred): 

    """

    Quite simply, the IoU metric measures the number of pixels common between the y_true
    and y_pred masks divided by the total number of pixels present across both masks.

    This approach just gives information about overlapping classes, e.g, if there is a pixel
    of class 3 in the true mask, and there is a pixel of class 2 in the same position in the
    pred mask, IoU will consider them the same, i.e binary IoU , so we need to use accuracy
    metric along with IoU

    labels: ground truth
    logits: last layer of the model, with softmax

    """
    y_true = tf.argmax(y_true , axis=-1)
    y_pred = tf.argmax(y_pred , axis=-1)
    y_true = y_true > 0
    y_pred = y_pred > 0
    intersection = tf.logical_and(y_true, y_pred)
    union = tf.logical_or(y_true, y_pred)
    intersection = tf.count_nonzero(intersection)
    union = tf.count_nonzero(union)
    iou_score = intersection / union
    return iou_score

def loss_custom_function(y_true, y_fake, weights):
    """ 
    Our crossentropy method,  with custom weights for each class, it had some
    computational problems, we need to use reduce mean instead of reduce sum 
    and use an 'epsilon' to avoid not a number problems
    """
    loss = tf.losses.softmax_cross_entropy(y_true, y_fake , _w)

    loss = y_true * tf.log(y_fake)
    loss = loss * _w
    loss = -tf.reduce_sum(loss)
    return loss


def my_loss_ultimatum(y_true, y_fake, gamma=2):
    """
    Focal loss , we found this function in this paper:
    https://arxiv.org/abs/1708.02002
    
    This function speeded up our training but it was only tested in Unet and 
    SegNet without pre-trained weights
    """
    
    return -tf.reduce_sum(tf.pow(1. - y_fake, gamma) * y_true * tf.log(y_fake)) #Focal Loss
