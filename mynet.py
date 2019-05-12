import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import Model
from keras import layers
from keras.layers import Input, Conv2D, MaxPooling2D,AveragePooling2D, Dropout, concatenate, Dense, Activation, SeparableConv2D
from keras.layers import GlobalAveragePooling2D, Reshape, multiply, UpSampling2D, Concatenate
from keras.optimizers import SGD, rmsprop, Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
#from keras.regularizers import l2,activity_l2
from keras.regularizers import l2 , l1
#from keras import backend as keras
import keras

def UpSampling(x):
    # 上采样
    y = UpSampling2D(size = (2,2))(x)
    return y

def GFinception(x):
    inception_1x1_1 = Conv2D(32,1,border_mode='same',activation='relu',name='inception_3a/1x1',W_regularizer=l2(0.0002),activity_regularizer=l1(0.01))(x)
    inception_3x3_2 = Conv2D(32,1,border_mode='same',activation='relu',name='inception_3a/3x3_reduce',W_regularizer=l2(0.0002),activity_regularizer=l1(0.01))(x)
    inception_3x3_3 = Conv2D(64,3,border_mode='same',activation='relu',name='inception_3a/3x3',W_regularizer=l2(0.0002),activity_regularizer=l1(0.01))(inception_3x3_2)
    inception_5x5_4 = Conv2D(64,3,border_mode='same',activation='relu',name='inception_3a/5x5_reduce',W_regularizer=l2(0.0002),activity_regularizer=l1(0.01))(x)
    inception_5x5_5 = Conv2D(128,5,border_mode='same',activation='relu',name='inception_3a/5x5',W_regularizer=l2(0.0002),activity_regularizer=l1(0.01))(inception_5x5_4)
    #inception_pool_1 = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='inception_3a/pool')(x)
    
    inception_pool_1cov = Conv2D(256,7,border_mode='same',activation='relu',name='inception_3a/pool_proj',W_regularizer=l2(0.0002))(x)
    drop1 = Dropout(0.5)(inception_pool_1cov)
    #y = merge([inception_1x1_1,inception_3x3_3,inception_5x5_5,drop1],mode='concat',concat_axis=1,name='inception/output')
    y = concatenate([inception_1x1_1,inception_3x3_3,inception_5x5_5,drop1], axis = 3)
    return y    
def ASPP(x, rate, filters):
    # Atrous Spatial Pyramid Pooling
    DeepConv1 = SeparableConv2D(filters,kernel_size=(3, 3), strides=(1, 1), dilation_rate=rate[0], padding='same', use_bias=False)(x)
    DeepConv2 = SeparableConv2D(filters,kernel_size=(3, 3), strides=(1, 1), dilation_rate=rate[1], padding='same', use_bias=False)(x)
    DeepConv3 = SeparableConv2D(filters,kernel_size=(3, 3), strides=(1, 1), dilation_rate=rate[2], padding='same', use_bias=False)(x)
    y = Concatenate()([x, DeepConv1, DeepConv2, DeepConv3])
    return y

def myASPP(x, rate, filters):
    # my Atrous Spatial Pyramid Pooling
    DeepConv0 = SeparableConv2D(filters,kernel_size=(3, 3), strides=(1, 1), dilation_rate=rate[0], padding='same', use_bias=False)(x)
    DeepConv1 = SeparableConv2D(filters,kernel_size=(3, 3), strides=(1, 1), dilation_rate=rate[1], padding='same', use_bias=False)(x)
    DeepConv2 = SeparableConv2D(filters,kernel_size=(3, 3), strides=(1, 1), dilation_rate=rate[2], padding='same', use_bias=False)(x)
    DeepConv3 = SeparableConv2D(filters,kernel_size=(3, 3), strides=(1, 1), dilation_rate=rate[3], padding='same', use_bias=False)(x)
    
    y = Concatenate()([x, DeepConv0, DeepConv1, DeepConv2, DeepConv3])
    return y
    
def SEnet(x, fc_size, filters_output):
    # Squeeze And Excitation
    squeeze = GlobalAveragePooling2D()(x)
    excitation = Dense(units=fc_size // 1)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=fc_size)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape((1, 1, fc_size))(excitation)
    scale = multiply([x, excitation])
    
    # y = Conv2D(filters_output, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(scale)
    y = SeparableConv2D(filters_output,kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(scale)
    return y

def myUnet(classes, input_size, epochs, batch_size, LR,
          Falg_summary=False, Falg_plot_model=False, pretrained_weights = None):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv1_1')(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv1_2')(conv1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
    #pool1 = SeparableConv2D(32,kernel_size=(3, 3), strides=(2, 2), dilation_rate=1, padding='same', use_bias=False)(drop1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv2_1')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv2_2')(conv2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    #pool2 = SeparableConv2D(64,kernel_size=(3, 3), strides=(2, 2), dilation_rate=1, padding='same', use_bias=False)(drop2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv3_1')(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv3_2')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    #pool3 = SeparableConv2D(128,kernel_size=(3, 3), strides=(2, 2), dilation_rate=1, padding='same', use_bias=False)(drop3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv4_1')(pool3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv4_2')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    #pool4 = SeparableConv2D(256,kernel_size=(3, 3), strides=(2, 2), dilation_rate=1, padding='same', use_bias=False)(drop4)
    
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv5_1')(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv5_2')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling(drop5))
    ASPP1 = myASPP(drop4, rate=[6, 12, 18, 24], filters=256)
    ASPP1_SE = SEnet(ASPP1, fc_size=256*5, filters_output=256)
    merge6 = concatenate([ASPP1_SE,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling(conv6))
    ASPP2 = myASPP(drop3, rate=[6, 12, 18, 24], filters=128)
    ASPP2_SE = SEnet(ASPP2, fc_size=128*5, filters_output=128)
    merge7 = concatenate([ASPP2_SE,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling(conv7))
    ASPP3 = myASPP(drop2, rate=[6, 12, 18, 24], filters=64)
    ASPP3_SE = SEnet(ASPP3, fc_size=64*5, filters_output=64)
    merge8 = concatenate([ASPP3_SE,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling(conv8))
    ASPP4 = myASPP(drop1, rate=[6, 12, 18, 24], filters=32)
    ASPP4_SE = SEnet(ASPP4, fc_size=32*5, filters_output=32)
    merge9 = concatenate([ASPP4_SE,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
    
    opt = rmsprop(lr=LR, decay=LR / epochs) 
    #opt = SGD(lr=LR, momentum=0.9, decay=LR / epochs, nesterov=False)
    #opt = SGD(lr=LR, momentum=0.9, decay=LR / epochs)
    #opt = Adam(lr=LR, beta_1=0.9, beta_2=0.999, decay=LR / epochs, epsilon=1e-08)
    #opt = Adam(lr = LR)
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    if Falg_summary:
        model.summary()
    if Falg_plot_model:
        keras.utils.plot_model(model, to_file='logs/mynet/model.png', show_shapes=True, show_layer_names=True)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet_pse(classes, input_size, epochs, batch_size, LR,
          Falg_summary=False, Falg_plot_model=False, pretrained_weights = None):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #pool1 = SeparableConv2D(64,kernel_size=(3, 3), strides=(2, 2), dilation_rate=1, padding='same', use_bias=False)(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #pool2 = SeparableConv2D(128,kernel_size=(3, 3), strides=(2, 2), dilation_rate=1, padding='same', use_bias=False)(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #pool3 = SeparableConv2D(256,kernel_size=(3, 3), strides=(2, 2), dilation_rate=1, padding='same', use_bias=False)(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    #pool4 = SeparableConv2D(512,kernel_size=(3, 3), strides=(2, 2), dilation_rate=1, padding='same', use_bias=False)(conv4)
    
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    x = Dropout(0.5)(conv5)
    # =============================================================================
    #Middle flow
    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 1)

        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = layers.Conv2D(1024, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv1')(x)
        x = layers.BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = layers.Conv2D(1024, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv2')(x)
        x = layers.BatchNormalization(name=prefix + '_sepconv2_bn')(x)

        x = layers.add([x, residual])

    # =============================================================================
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(x))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    
    ASPP = myASPP(conv9, rate=[6, 12, 18, 24], filters=64)
    ASPP_SE = SEnet(ASPP, fc_size=64*5, filters_output=64)
    
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ASPP_SE)
    conv11 = Conv2D(1, 1, activation = 'sigmoid')(conv10)

    model = Model(input = inputs, output = conv11)
    
    #opt = rmsprop(lr=0.0003, decay=1e-6) 
    #opt = SGD(lr=0.001, momentum=0.9)
    #opt = SGD(lr=LearningRate, momentum=0.9, decay=LearningRate / epochs)
    #opt = Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt = Adam(lr = 1e-4)
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    if Falg_summary:
        model.summary()
    if Falg_plot_model:
        keras.utils.plot_model(model, to_file='logs/Unet/model.png', show_shapes=True, show_layer_names=True)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
    
def myXceptFCN(classes, input_size, epochs, batch_size, LR,
             Falg_summary=False, Falg_plot_model=False,
             include_top=True, pooling='avg', pretrained_weights = None):
    
    inputs = Input(input_size)
    # =============================================================================
    # Entry flow
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False, name='block1_conv1')(inputs)
    x = layers.BatchNormalization(name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    block1_res = layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False, name='block1_res')(x)
    block1_res = layers.BatchNormalization()(block1_res)

    x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = layers.BatchNormalization(name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = layers.BatchNormalization(name='block2_sepconv2_bn')(x)

    #x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = layers.SeparableConv2D(128, (3, 3), strides=(2, 2), padding='same', use_bias=False, name='block2_atrous1')(x)
    x = layers.add([x, block1_res])

    block2_res = layers.Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False, name='block2_res')(x)
    block2_res = layers.BatchNormalization()(block2_res)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = layers.BatchNormalization(name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = layers.BatchNormalization(name='block3_sepconv2_bn')(x)

    #x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = layers.SeparableConv2D(256, (3, 3), strides=(2, 2), padding='same', use_bias=False, name='block3_atrous1')(x)
    x_asppInput1 = layers.add([x, block2_res])

    block3_res = layers.Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False, name='block3_res')(x_asppInput1)
    block3_res = layers.BatchNormalization()(block3_res)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = layers.BatchNormalization(name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = layers.BatchNormalization(name='block4_sepconv2_bn')(x)

    #x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = layers.SeparableConv2D(256, (3, 3), strides=(2, 2), padding='same', use_bias=False, name='block4_atrous1')(x)
    x = layers.add([x, block3_res])
    # =============================================================================
    # Middle flow
    for i in range(2):
        residual = x
        prefix = 'block' + str(i + 5)

        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = layers.SeparableConv2D(256, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv1')(x)
        x = layers.BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = layers.SeparableConv2D(256, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv2')(x)
        x = layers.BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = layers.SeparableConv2D(256, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv3')(x)
        x = layers.BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])
    # =============================================================================
    # Exit flow
    residual = layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = layers.BatchNormalization(name='block13_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block13_sepconv2_act')(x)
    x = layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = layers.BatchNormalization(name='block13_sepconv2_bn')(x)

    #x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    x = layers.SeparableConv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='block13_atrous1')(x)
    x = layers.add([x, residual])

    x = layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = layers.BatchNormalization(name='block14_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv1_act')(x)
    
    x = layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = layers.BatchNormalization(name='block14_sepconv2_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv2_act')(x)

    x = layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block14_sepconv3')(x)
    x = layers.BatchNormalization(name='block14_sepconv3_bn')(x)
    x_asppInput2 = layers.Activation('relu', name='block14_sepconv3_act')(x)

    
    aspp1 = myASPP(x_asppInput1, [1, 2, 3, 4])
    aspp1_up = UpSampling2D(size = (2,2))(aspp1)
    aspp1_SE = SEnet(aspp1_up, 1280)
    
    aspp2 = myASPP(x_asppInput2, [1, 2, 3, 4])
    aspp2_up = UpSampling2D(size = (4,4))(aspp2)
    aspp2_SE = SEnet(aspp2_up, 1280)
    
    x = Concatenate()([aspp1_SE, aspp2_SE])
    x = UpSampling2D(size = (4,4))(x)
    
    x = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(1, 1, activation = 'sigmoid')(x)
    #x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    #x = layers.Dense(1, activation='sigmoid', name='predictions')(x)

    # =============================================================================
    # Create model.
    model = Model(inputs, x, name='xception-FCN')
    
    #opt = rmsprop(lr=LR, decay=1e-6) 
    opt = SGD(lr=LR, momentum=0.9, decay=LR / epochs, nesterov=False)
    #opt = SGD(lr=LR, momentum=0.9, decay=LR / epochs)
    #opt = Adam(lr=LR, beta_1=0.9, beta_2=0.999, decay=0, epsilon=1e-08)
    #opt = Adam(lr = LR)
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    if Falg_summary:
        model.summary()
    if Falg_plot_model:
        keras.utils.plot_model(model, to_file='logs/XceptFCN/model.png', show_shapes=True, show_layer_names=True)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
        
    return model




def unet3(classes, input_size, epochs, batch_size, LR,
          Falg_summary=False, Falg_plot_model=False, pretrained_weights = None):
    inputs = Input(input_size)
    
#NO.1    
    #第一组两个卷积
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    #pool1 = GFinception(pool1)
    
    #第二组两个卷积
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #第三组两个卷积
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #第四组两个卷积
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    #pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

#NO.2 

#第一组两个卷积
    conv1_2 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1_2 = Conv2D(64, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_2)
    pool1_2 = AveragePooling2D(pool_size=(2, 2))(conv1_2)
    
    #pool1_2 = GFinception(pool1)
    
    #第二组两个卷积
    conv2_2 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1_2)
    conv2_2 = Conv2D(128, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_2)
    pool2_2 = AveragePooling2D(pool_size=(2, 2))(conv2_2)
    #第三组两个卷积
    conv3_2 = Conv2D(256, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2_2)
    conv3_2 = Conv2D(256, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_2)
    pool3_2 = AveragePooling2D(pool_size=(2, 2))(conv3_2)
    #第四组两个卷积
    conv4_2 = Conv2D(512, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3_2)
    conv4_2 = Conv2D(512, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
    drop4_2 = Dropout(0.5)(conv4_2)
    #pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)


#NO.3 


#第一组两个卷积
    conv1_3 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1_3 = Conv2D(64, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_3)
    pool1_3 = AveragePooling2D(pool_size=(2, 2))(conv1_3)
    
    #pool1_3 = GFinception(pool1)
    
    #第二组两个卷积
    conv2_3 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1_3)
    conv2_3 = Conv2D(128, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_3)
    pool2_3 = MaxPooling2D(pool_size=(2, 2))(conv2_3)
    #第三组两个卷积
    conv3_3 = Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2_3)
    conv3_3 = Conv2D(256, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_3)
    pool3_3 = AveragePooling2D(pool_size=(2, 2))(conv3_3)
    #第四组两个卷积
    conv4_3 = Conv2D(512, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3_3)
    conv4_3 = Conv2D(512, 5, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_3)
    drop4_3 = Dropout(0.5)(conv4_3)
    #pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    merge_add_3 = concatenate([drop4,drop4_2,drop4_3], axis = 3)




    #最底层两个卷积
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_add_3)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    #第一个上采样 和concate
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (1,1))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    #第二个上采样 和concate
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    #第三个上采样 和concate
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    #第四个上采样 和concate
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name="8888")(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
    
    #opt = rmsprop(lr=0.0003, decay=1e-6) 
    #opt = SGD(lr=0.001, momentum=0.9)
    #opt = SGD(lr=LearningRate, momentum=0.9, decay=LearningRate / epochs)
    #opt = Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt = Adam(lr = 1e-4)
    model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    if Falg_summary:
        model.summary()
    if Falg_plot_model:
        keras.utils.plot_model(model, to_file='logs/Unet/model.png', show_shapes=True, show_layer_names=True)
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


















