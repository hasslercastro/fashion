from keras.models import Model, load_model, Sequential
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, UpSampling2D, Permute
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.optimizers import Adam

def simple_block(_input , kernel, feature_maps, decoder=False):
    X = Conv2D(feature_maps , (kernel,kernel) , strides=(1,1) ,padding='same', activation='relu')(_input)
    X = BatchNormalization()(X)
    X = Conv2D(feature_maps , (kernel,kernel) , strides=(1,1) ,padding='same', activation='relu')(X)
    X = BatchNormalization()(X)
    if decoder:
        X = UpSampling2D(size=(2,2))(X)
        return X
        
    X = MaxPooling2D(pool_size=(2,2))(X)
    return X


def complex_block(_input, kernel, feature_maps, decoder = False):
    X = Conv2D(feature_maps , (kernel,kernel) , strides=(1,1) ,padding='same', activation='relu')(_input)
    X = BatchNormalization()(X)
    X = Conv2D(feature_maps , (kernel,kernel) , strides=(1,1) ,padding='same', activation='relu')(X)
    X = BatchNormalization()(X)
    X = Conv2D(feature_maps , (kernel,kernel) , strides=(1,1) ,padding='same', activation='relu')(X)
    X = BatchNormalization()(X)
    if decoder:
        X = UpSampling2D(size=(2,2))(X)
        return X
        
    X = MaxPooling2D(pool_size=(2,2))(X)
    return X

def final_block(_input, kernel, feature_maps, classes , img_w, img_h,channels):
    X = Conv2D(feature_maps, (kernel, kernel), strides=(1, 1), padding='same', activation='relu')(_input)  
    X = BatchNormalization()(X)
    X = Conv2D(feature_maps, (kernel, kernel), strides=(1, 1), padding='same', activation='relu')(X)  
    X = BatchNormalization()(X)
    X = Conv2D(classes, (1, 1), strides=(1, 1), padding='same')(X)
    return X

def seg_net(input_shape , classes = 47 , kernel = 3, fmap = 64):  
    
    _input = Input(input_shape)
    X = simple_block(_input , kernel,  fmap)
    #start encoder         
    #ouput size (128,128)  
    X = simple_block(X , kernel , fmap*2) 
    #ouput size (64,64)  
    X = complex_block(X, kernel, fmap*4)
    #ouput size (32,32)  
    X = complex_block(X, kernel, fmap*8)
    #ouput size (16,16)  
    X = complex_block(X, kernel, fmap*8)
    #ouput size (8,8)  
    #end encoder, begin decoder
    X = UpSampling2D(size=(2,2))(X)  
    #ouput size (16,16)  
    X = complex_block(X , kernel, fmap*8, decoder=True)
    #ouput size (32,32)  
    X = complex_block(X , kernel, fmap*8, decoder=True)
    #ouput size (64,64)  
    X = complex_block(X , kernel, fmap*4, decoder=True)
    #ouput size (128,128)  
    X = simple_block(X , kernel, fmap*2, decoder=True)
    #ouput size (256,256)    
    X = final_block(X , kernel, fmap, classes, *input_shape)
    X = Activation('softmax')(X)

    model = Model(inputs=_input , outputs=X)

    return model
