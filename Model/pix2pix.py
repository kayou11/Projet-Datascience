# -*- coding: utf-8 -*-
"""Pierre - Pix2Pix with classes.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tJV5QJS-sUbZ_5_zcWeVSKaPCr7hOzb1
"""

import numpy as np 
import pandas as pd 
import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from tqdm import tqdm_notebook as tqdm

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import datetime
import sys
import os
import imageio
from PIL import ImageChops
from scipy.linalg import norm
from scipy import sum, average
from skimage.metrics import structural_similarity as ssim

import math, operator
#sys.path.append("/content/drive/My Drive/Projet DataScience") #Path Pierre
sys.path.append("/content/drive/My Drive/CESI/Projets A5/Data Science/Projet DataScience") #Path Kayou
from Pipeline.Degradation import UglyImage

class DataLoader():
  def __init__(self, img_res=(128,128)):
    self.img_res = img_res
    #Path Pierre
    #self.train_path_files = '/content/drive/My Drive/Projet DataScience/Data/Train/dataset_clean_degraded'
    #self.val_path_files = '/content/drive/My Drive/Projet DataScience/Data/Val/'

    #Path Kayou
    #self.train_path_files = '/content/drive/My Drive/CESI/Projets A5/Data Science/Projet DataScience/Data/Train/dataset_clean_degraded'
    #self.val_path_files = '/content/drive/My Drive/CESI/Projets A5/Data Science/Projet DataScience/Data/Val'

    #Path workflow
    self.train_path_files = '/content/Val'
    self.val_path_files = '/content/Val'

  def intersection(self, lst1, lst2): 
    result=[]
    for i in lst2:
        if isinstance(i,list):
            result.append(intersect(lst1,i))
        else:
            if i in lst1:
                 result.append(i)
    return result

  def load_data(self, batch_size=1, is_val=True):
    """
    Return couples of images to visualize progress of the networks after epochs
    """
    
    path_files = self.train_path_files if not is_val else self.val_path_files

    clean_images = []
    degraded_images = []
    
    files = os.listdir(path_files + '/clean/')
    batch_images = np.random.choice(files, size=batch_size)

    for image in batch_images:
      clean = self.imread(path_files + '/clean/' + image)
      degraded = self.imread(path_files + '/degraded/' + image)

      # Decrease resolution
      clean = transform.resize(clean, self.img_res)
      degraded = transform.resize(degraded, self.img_res)

      clean_images.append(clean)
      degraded_images.append(degraded)

    #normalizing images
    clean_images = np.array(clean_images)/127.5 - 1.
    degraded_images = np.array(degraded_images)/127.5 -1.

    return clean_images, degraded_images

  def load_batch(self, batch_size=1, is_val=False):
    """
    Same function as load_data except for the fact that is used during training to load image in batches
    """
    
    path_files = self.train_path_files if not is_val else self.val_path_files

    n_batches = batch_size
    files = os.listdir(path_files + '/clean/')
    #files_clean = os.listdir(path_files + '/clean/')
    #files_degraded = os.listdir(path_files + '/degraded/')
    #files = self.intersection(files_clean, files_degraded) 

    #ugly = UglyImage(path=path_files + '/clean/', image_size=self.img_res)

    print("Load Batch")
    for i in tqdm(range(n_batches)):
      batch = files[i*batch_size:(i+1)*batch_size]
      #batch = np.random.choice(files, size=batch_size)

      clean_images = []
      degraded_images = []

      for image in batch:
        #clean_image_path = str(path_files + '/clean/' + image)
        #degraded, clean = ugly.uglifyImage(clean_image_path)
        clean = self.imread(path_files + '/clean/' + image)
        degraded = self.imread(path_files + '/degraded/' + image)
      
        # decrease resolution
        clean = transform.resize(clean, self.img_res)
        degraded = transform.resize(degraded, self.img_res)

        clean_images.append(clean)
        degraded_images.append(degraded)

      #normalizing images
      clean_images = np.array(clean_images)/127.5 - 1.
      degraded_images = np.array(degraded_images)/127.5 -1.

      yield clean_images, degraded_images

  def imread(self, path):
    return imageio.imread(path).astype(np.float)

class Pix2Pix():
  def __init__(self, img_rows=128, img_cols=128, channels=3):
    # Input shape
    self.img_rows = img_rows
    self.img_cols = img_cols
    self.channels = channels

    self.img_shape = (self.img_rows, self.img_cols, self.channels)

    # Configure DataLoader
    self.data_loader = DataLoader(img_res=(self.img_rows, self.img_cols))

    # Calculate output shape of D (PatchGAN)
    patchrows = int(self.img_rows / 2**4)
    patchcols = int(self.img_cols / 2**4)
    self.disc_patch = (patchrows, patchcols, 1)

    # Number of filters in the first layer of G and D
    self.gf = 64
    self.df = 64

    optimizer = Adam(0.0002, 0.5)

    self.generator_weights_filepath = 'weights_generator.h5'
    self.discriminator_weights_filepath = 'weights_discriminator.h5'

    # Build and compile the discriminator
    self.discriminator = self.build_discriminator()
    self.discriminator.compile(loss='mse',
                optimizer=optimizer,
                metrics=['accuracy'])

    # Build the generator
    self.generator = self.build_generator()

    # Input images and their conditioning images
    clean_image = Input(shape=self.img_shape)
    degraded_image = Input(shape=self.img_shape)

    # By conditioning on degraded_image generate a fake version of clean_image
    fake_clean_image = self.generator(degraded_image)

    # For the combined model we will only train the generator
    self.discriminator.trainable = False

    # Discriminators determines validity of translated images / condition pairs
    valid = self.discriminator([fake_clean_image, degraded_image])

    self.combined = Model(inputs=[clean_image, degraded_image], outputs=[valid, fake_clean_image])
    self.combined.compile(loss=['mse', 'mae'],
                                  loss_weights=[1, 100],
                                  optimizer=optimizer)
    
  
  def build_generator(self):
    """
    U-Net Generator (to generate image)
    """

    def conv2d(input_layer, filters, f_size=4, bn=True):
      """
      Layers used during downsampling
      """
      d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(input_layer)
      d = LeakyReLU(alpha=0.2)(d)
      
      if bn:
        d = BatchNormalization(momentum=0.8)(d)
      
      return d

    def deconv2d(input_layer, skip_input, filters, f_size=4, dropout_rate=0):
      """
      Layers used during downsampling
      """
      u = UpSampling2D(size=2)(input_layer)
      u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)

      if dropout_rate:
        u = Dropout(dropout_rate)(u)
      
      u = BatchNormalization(momentum=0.8)(u)
      u = Concatenate()([u, skip_input])

      return u

    #Image Input
    d0 = Input(shape=self.img_shape)

    #Downsampling
    d1 = conv2d(d0, self.gf, bn=False)
    d2 = conv2d(d1, self.gf*2)
    d3 = conv2d(d2, self.gf*4)
    d4 = conv2d(d3, self.gf*8)
    d5 = conv2d(d4, self.gf*8)
    d6 = conv2d(d5, self.gf*8)
    d7 = conv2d(d6, self.gf*8)

    # Upsampling
    u1 = deconv2d(d7, d6, self.gf*8)
    u2 = deconv2d(u1, d5, self.gf*8)
    u3 = deconv2d(u2, d4, self.gf*8)
    u4 = deconv2d(u3, d3, self.gf*4)
    u5 = deconv2d(u4, d2, self.gf*2)
    u6 = deconv2d(u5, d1, self.gf)

    u7 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

    return Model(d0, output_img)

  
  def build_discriminator(self):
  
    def d_layer(input_layer, filters, f_size=4, bn=True):
      """
      Discriminator layer
      """
      d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(input_layer)
      d = LeakyReLU(alpha=0.2)(d)

      if bn:
        d = BatchNormalization(momentum=0.8)(d)
      return d

    clean_image = Input(shape=self.img_shape)
    degraded_image = Input(shape=self.img_shape)

    # Concatenate image and conditioning image by chanels to produce input
    combined_imgs = Concatenate(axis=-1)([clean_image, degraded_image])

    d1 = d_layer(combined_imgs, self.df, bn=False)
    d2 = d_layer(d1, self.df*2)
    d3 = d_layer(d2, self.df*4)
    d4 = d_layer(d3, self.df*8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model([clean_image, degraded_image], validity)

  def compare_images(self, img1, img2):
    # calculate the difference and its norms
    diff = img1 - img2  # elementwise for scipy arrays
    m_norm = sum(abs(diff))  # Manhattan norm or L1
    #z_norm = norm(diff.ravel(), 0)  # Zero norm
    e_norm = np.sqrt(sum(diff**2)) # Euclidean norm or L2
    return (m_norm, e_norm)

  def ssim(self, img1, img2):
    similarity = ssim(img1, img2)
    return similarity

  def to_grayscale(self, arr):
    "If arr is a color image (3D array), convert it to grayscale (2D array)."
    if len(arr.shape) == 3:
      return average(arr, -1)  # average over the last axis (color channels)
    else:
      return arr

  def evaluate(self):
    '''Return improvment message if quality is better on validation set at the end of the training'''
    clean_images, degraded_images = self.data_loader.load_data(batch_size=10, is_val=True)
    fake_clean_images = self.generator.predict(degraded_images)

    improve_message = ""

    l1_all_diff = []
    l1_is_quality_better = []
    
    l2_all_diff = []
    l2_is_quality_better = []

    for i in range(0, len(clean_images)):
      clean_image = self.to_grayscale(clean_images[i].astype(float))
      degraded_image = self.to_grayscale(degraded_images[i].astype(float))
      fake_clean_image = self.to_grayscale(fake_clean_images[i].astype(float))

      l1_distance_clean_degraded, l2_distance_clean_degraded = self.compare_images(degraded_image, clean_image)
      l1_distance_predict_clean, l2_distance_predict_clean = self.compare_images(fake_clean_image, clean_image)

      l1_diff = float(l1_distance_clean_degraded)/float(l1_distance_predict_clean) * 100
      l2_diff = float(l2_distance_clean_degraded)/float(l2_distance_predict_clean) * 100

      l1_all_diff.append(l1_diff)
      
      if l1_distance_predict_clean < l1_distance_clean_degraded:
        l1_is_quality_better.append(True)
      else:
        l1_is_quality_better.append(False)
      
      if l2_distance_predict_clean < l2_distance_clean_degraded:
        l2_is_quality_better.append(True)
      else:
        l2_is_quality_better.append(False)

      l2_all_diff.append(l2_diff)

    average_l1_diff = sum(l1_all_diff) / len(l1_all_diff)
    average_l2_diff = sum(l2_all_diff) / len(l2_all_diff)

    l1_average_is_quality_better = "détériorée"
    if (sum(l1_is_quality_better)/len(l1_is_quality_better) >= 0.5):
      l1_average_is_quality_better = "améliorée"

    l2_average_is_quality_better = "détériorée"
    if (sum(l2_is_quality_better)/len(l2_is_quality_better) >= 0.5):
      l2_average_is_quality_better = "améliorée"

    improve_message += ("Selon L1, les images du jeu de validation se sont en moyenne " + l1_average_is_quality_better + " de " + str(round(average_l1_diff,2)) + "% \n")
    improve_message += ("Selon L2, les images du jeu de validation se sont en moyenne " + l2_average_is_quality_better + " de " + str(round(average_l2_diff,2)) + "% \n")

    return improve_message
  
  def validate(self, clean_images, degraded_images, fake_clean_images):
    '''Return improvment message if quality is better on 3 random images in validation set at the end of the epoch interval'''
    
    improve_message = ""

    for i in range(0, len(clean_images)):
      #print(clean_images[i])
      clean_image = self.to_grayscale(clean_images[i].astype(float))
      degraded_image = self.to_grayscale(degraded_images[i].astype(float))
      fake_clean_image = self.to_grayscale(fake_clean_images[i].astype(float))
      
      ssim_original = self.ssim(clean_image, clean_image)
      ssim_clean_degraded = self.ssim(clean_image, degraded_image)
      ssim_predict_clean = self.ssim(clean_image, fake_clean_image)
      
      l1_distance_clean_degraded, l2_distance_clean_degraded = self.compare_images(degraded_image, clean_image)
      l1_distance_predict_clean, l2_distance_predict_clean = self.compare_images(fake_clean_image, clean_image)

      l1_diff = float(l1_distance_clean_degraded)/float(l1_distance_predict_clean) * 100
      l2_diff = float(l2_distance_clean_degraded)/float(l2_distance_predict_clean) * 100

      l1_is_quality_better = "détériorée"
      l2_is_quality_better = "détériorée"

      if l1_distance_predict_clean < l1_distance_clean_degraded:
        l1_is_quality_better = "améliorée"

      if l2_distance_predict_clean < l2_distance_clean_degraded:
        l2_is_quality_better = "améliorée"
      
      #improve_message += ("\nSelon L1, l'image " + str(i) + ": s'est "+ l1_is_quality_better + " de " + str(round(l1_diff,2)) + " % \n")
      #improve_message += ("Selon L2, l'image " + str(i) + ": s'est "+ l2_is_quality_better + " de " + str(round(l2_diff,2)) + " % \n")
      
      improve_message += ("\nImage " + str(i) + " SSIM original : " + str(ssim_original) + "\n")
      improve_message += ("Image " + str(i) + " SSIM clean-degraded : " + str(ssim_clean_degraded) + "\n")
      improve_message += ("Image " + str(i) + " SSIM predict-clean : " + str(ssim_predict_clean) + "\n\n")

    return improve_message
  
  def train(self, epochs, batch_size=1, show_interval=10):
    start_time = datetime.datetime.now()

    generator_best_loss = None
    discriminator_best_loss = None

    # Adversarial loss ground truths
    valid = np.ones((batch_size,) + self.disc_patch)
    fake = np.zeros((batch_size,) + self.disc_patch)

    for epoch in range(epochs):
        for batch_i, (clean_images, degraded_images) in enumerate(self.data_loader.load_batch(batch_size)):

            # Train disciminator

            # Condition on degraded_images and translated version
            fake_clean_images = self.generator.predict(degraded_images)

            # Train the disciminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch([clean_images, degraded_images], valid)
            d_loss_fake = self.discriminator.train_on_batch([fake_clean_images, degraded_images], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # Train generator
            g_loss = self.combined.train_on_batch([clean_images, degraded_images], [valid, clean_images])
            elapsed_time = datetime.datetime.now() - start_time
        
        if epoch % show_interval == 0:
          # Plot the progress
          print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs, d_loss[0], 100*d_loss[1], g_loss[0], elapsed_time))

          # If at show interval => show generated image samples
          self.show_images(epoch, batch_i)

        # Save models
        for model, best_loss, loss, model_weights_filepath in [[self.generator, generator_best_loss, g_loss[0], self.generator_weights_filepath], [self.discriminator, discriminator_best_loss, d_loss[0], self.discriminator_weights_filepath]]:
          if best_loss == None or best_loss > loss:
            best_loss = loss
            model.save_weights(model_weights_filepath)
            
          else:
            pass
    print(self.evaluate())

  def show_images(self, epoch, batch_i):
        
    r, c = 3, 3

    clean_images, degraded_images = self.data_loader.load_data(batch_size=3, is_val=True)
    fake_clean_images = self.generator.predict(degraded_images)

    gen_imgs = np.concatenate([degraded_images, fake_clean_images, clean_images])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    titles = ['Input', 'Output', 'Ground Truth']
    fig, axs = plt.subplots(r, c)
    fig.set_size_inches(12, 12)
    cnt = 0
    for i in range(r):
      for j in range(c):
        axs[i,j].imshow(gen_imgs[cnt])
        axs[i,j].set_title(titles[i])
        axs[i,j].axis('off')
        cnt += 1
    print(self.validate(clean_images, degraded_images, fake_clean_images))
    plt.show()
    plt.close()
