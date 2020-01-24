# -*- coding: utf-8 -*-
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
import skimage
from PIL import Image
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.util import crop, pad
from skimage.morphology import label
from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb
from sklearn.model_selection import train_test_split

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model, load_model,Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, UpSampling2D, RepeatVector, Reshape
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

import tensorflow as tf
import glob

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

def get_data():

    IMG_WIDTH = 256
    IMG_HEIGHT = 256
    IMG_CHANNELS = 3
    INPUT_SHAPE=(IMG_HEIGHT, IMG_WIDTH, 1)
    TRAIN_PATH = '/content/Train/clean/'
    train_ids = next(os.walk(TRAIN_PATH))[2]

    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    filenames = os.listdir(TRAIN_PATH)
    filenames = filenames[:99]

    for i in range(len(filenames)):  
        img = imread(TRAIN_PATH + filenames[i])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[i] =img
    X_train = X_train.astype('float32') / 255.
    X_train, X_test = train_test_split(X_train, test_size=10, random_state=seed)

    return X_train

def Colorize():
    embed_input = Input(shape=(1000,))
    
    #Encoder
    encoder_input = Input(shape=(256, 256, 1,))
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same',strides=1)(encoder_input)
    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
    encoder_output = Conv2D(128, (4,4), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same',strides=1)(encoder_output)
    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
    encoder_output = Conv2D(256, (4,4), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same',strides=1)(encoder_output)
    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
    encoder_output = Conv2D(256, (4,4), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
    
    #Fusion
    fusion_output = RepeatVector(32 * 32)(embed_input) 
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
    fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)
    
    #Decoder
    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (4,4), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(32, (2,2), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    return Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

def get_inception():
    inception = InceptionResNetV2(weights=None, include_top=True)
    #inception.load_weights('../content/Model/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
    inception.graph = tf.get_default_graph()
    return inception, inception.graph

def get_model():
    model = Colorize()
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    return model

def data_generator():
    # Image transformer
    datagen = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.2,
            rotation_range=20,
            horizontal_flip=True)
    return datagen


#Create embedding
def create_inception_embedding(grayscaled_rgb):

    inception, inception.graph = get_inception()

    def resize_gray(x):
        return resize(x, (299, 299, 3), mode='constant')
    grayscaled_rgb_resized = np.array([resize_gray(x) for x in grayscaled_rgb])
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed

#Generate training data
def image_a_b_gen(dataset, batch_size = 20):
    datagen = data_generator()
    for batch in datagen.flow(dataset, batch_size=batch_size):
        X_batch = rgb2gray(batch)
        grayscaled_rgb = gray2rgb(X_batch)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield [X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch


def get_parameters():
    learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 
                                                patience=3, 
                                                verbose=1, 
                                                factor=0.5,
                                                min_lr=0.00001)
    filepath = "/content/Projet-Datascience/Weights/Colorization_Model.h5"
    checkpoint = ModelCheckpoint(filepath,
                                save_best_only=True,
                                monitor='loss',
                                mode='min')

    model_callbacks = [learning_rate_reduction,checkpoint]

    return filepath, model_callbacks

class colorGen():

    def train(self,BATCH_SIZE, epochs):

        X_train = get_data()
        model = get_model()

        filepath, model_callbacks = get_parameters()

        model.fit_generator(image_a_b_gen(X_train,BATCH_SIZE),
                    epochs=epochs,
                    verbose=1,
                    steps_per_epoch=X_train.shape[0]/BATCH_SIZE,
                    allbacks=model_callbacks
                            )
        model.save(filepath)
        model.save_weights("/content/Weights/Colorization_Weights.h5")


    def test(self, X_test):

        sample = X_test
        model = load_model("/content/Projet-Datascience/Weights/Colorization_Model.h5")
        model.load_weights("/content/Projet-Datascience/Weights/Colorization_Weights.h5")

        color_me = gray2rgb(rgb2gray(sample))
        color_me_embed = create_inception_embedding(color_me)
        color_me = rgb2lab(color_me)[:,:,:,0]
        color_me = color_me.reshape(color_me.shape+(1,))

        output = model.predict([color_me, color_me_embed])
        output = output * 128

        decoded_imgs = np.zeros((len(output),256, 256, 3))

        for i in range(len(output)):
            cur = np.zeros((256, 256, 3))
            cur[:,:,0] = color_me[i][:,:,0]
            cur[:,:,1:] = output[i]
            decoded_imgs[i] = lab2rgb(cur)
            cv2.imwrite("img_"+str(i)+".jpg", lab2rgb(cur))


        plt.figure(figsize=(20, 6))
        for i in range(len(X_test)):
            # grayscale
            plt.subplot(3, 10, i + 1)
            plt.imshow(rgb2gray(X_test)[i].reshape(256, 256))
            plt.gray()
            plt.axis('off')
 
            # recolorization
            plt.subplot(3, 10, i + 1 +10)
            plt.imshow(decoded_imgs[i].reshape(256, 256,3))
            plt.axis('off')
    
            # original
            plt.subplot(3, 10, i + 1 + 20)
            plt.imshow(X_test[i].reshape(256, 256,3))
            plt.axis('off')
 
            plt.tight_layout()
            plt.show()
        
        return output
