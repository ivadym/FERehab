# IMPORTS ###########################################################################################################################################

import os
import datetime
import numpy as np
import pandas as pd
from scipy.misc import imsave
from io import BytesIO
from tensorflow.python.lib.io import file_io
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dropout, Concatenate, Activation, UpSampling2D, Conv2D, LeakyReLU
from utils.normalization import InstanceNormalization

# PARAMETERS ######################################################################################################################################

# Input shape
img_height, img_width = 48, 48 # FER-2013
channels    = 3
img_shape   = (img_height, img_width, channels)

# Loss weights
lambda_cycle    = 10.0  # Cycle-consistency loss

optimizer = Adam(0.0002, 0.5)

# Training
epochs          = 30000
batch_size      = 1
save_interval   = 100

# DATASETS ##########################################################################################################################################

# Folder where logs and models are stored
folder = "logs"

# Data paths
dataset_A = "./../FER-2013/train_neutral.npy"
dataset_B = "./../FER-2013/train_disgust.npy"

image_A = "./../FER-2013/eval_neutral.npy"
image_B = "./../FER-2013/eval_disgust.npy"

# DATA PREPARATION ##################################################################################################################################

def preprocess_input(x):    
    x /= 127.5
    x -= 1
    return x

# Function that reads the images from the csv file
    # dataset: Data path
def load_data(dataset):
    f = BytesIO(file_io.read_file_to_string(dataset, binary_mode = True))
    loaded_images = np.load(f)
    images = preprocess_input(loaded_images)

    return images

# GENERATOR ##########################################################################################################################################

def generator():
    """U-Net Generator"""

    def conv_2d_v0(layer_input, filters, f_size = 4):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size = f_size, strides = 2, padding = "same")(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d

    def deconv_2d_v0(layer_input, skip_input, filters, f_size = 4, dropout_rate = 0):
        """Layers used during upsampling"""
        u = UpSampling2D(size = 2)(layer_input)
        u = Conv2D(filters, kernel_size = f_size, strides = 1, padding = "same", activation = "relu")(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape = img_shape)

    # Downsampling
    d1 = conv_2d_v0(d0, 32)
    d2 = conv_2d_v0(d1, 32*2)
    d3 = conv_2d_v0(d2, 32*4)
    d4 = conv_2d_v0(d3, 32*8)

    # Upsampling
    u1 = deconv_2d_v0(d4, d3, 32*4)
    u2 = deconv_2d_v0(u1, d2, 32*2)
    u3 = deconv_2d_v0(u2, d1, 32)

    u4 = UpSampling2D(size = 2)(u3)
    output_img = Conv2D(channels, kernel_size = 4, strides = 1, padding = "same", activation = "tanh")(u4)

    return Model(d0, output_img)

# DISCRIMINATOR ######################################################################################################################################

def discrimiantor():
    
    def discriminator_layer(layer_input, filters, f_size = 4, normalization = True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size = f_size, strides = 2, padding = "same")(layer_input)
        d = LeakyReLU(alpha = 0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d

    img = Input(shape = img_shape)

    d1 = discriminator_layer(img, 64, normalization = False)
    d2 = discriminator_layer(d1, 64*2)
    d3 = discriminator_layer(d2, 64*4)
    d4 = discriminator_layer(d3, 64*8)

    validity = Conv2D(1, kernel_size = 4, strides = 1, padding = "same")(d4)

    return Model(img, validity)

# BUILDING AND COMPILING #############################################################################################################################

# Build and compile the discriminators
d_A = discrimiantor()
d_B = discrimiantor()
d_A.compile(
	loss 		= "mse",
    optimizer 	= optimizer,
    metrics 	= ["accuracy"])
d_B.compile(
	loss 		= "mse",
    optimizer 	= optimizer,
    metrics 	= ["accuracy"])

# Build and compile the generators
g_AB = generator()
g_BA = generator()
g_AB.compile(
	loss 		= "binary_crossentropy",
	optimizer 	= optimizer)
g_BA.compile(
	loss 		="binary_crossentropy",
	optimizer 	= optimizer)

# Input images from both domains
img_A = Input(shape = img_shape)
img_B = Input(shape = img_shape)

# Translate images to the other domain
fake_B = g_AB(img_A)
fake_A = g_BA(img_B)

# Translate images back to original domain
reconstr_A = g_BA(fake_B)
reconstr_B = g_AB(fake_A)

# For the combined model we will only train the generators (updates discriminators weights only when discriminator.fit() is called but not when gan.fit() is called)
d_A.trainable = False
d_B.trainable = False

# Discriminators determines validity of translated images
valid_A = d_A(fake_A)
valid_B = d_B(fake_B)
combined = Model(
    [img_A, img_B],
    [valid_A, valid_B, fake_B, fake_A, reconstr_A, reconstr_B])
combined.compile(
    loss 			= ["mse", "mse", "mae", "mae", "mae", "mae"],
    loss_weights	= [1, 1, 1, 1, lambda_cycle, lambda_cycle], optimizer = optimizer)

# SAVE FUNCITION ######################################################################################################################################

def check_file(filepath):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

def save_imgs(epoch):
    imgs_A = load_data(image_A)
    imgs_B = load_data(image_B)

    # Translate images to the other domain
    fake_B = g_AB.predict(imgs_A)
    fake_A = g_BA.predict(imgs_B)
    # Translate back to original domain
    reconstr_A = g_BA.predict(fake_B)
    reconstr_B = g_AB.predict(fake_A)

    gen_imgs = [np.squeeze(imgs_A, axis = 0), np.squeeze(fake_B, axis = 0), np.squeeze(reconstr_A, axis = 0),
        np.squeeze(imgs_B, axis = 0), np.squeeze(fake_A, axis = 0), np.squeeze(reconstr_B, axis = 0)]

    titles = ["Original_A", "Translated_A", "Reconstructed_A", "Original_B", "Translated_B", "Reconstructed_B"]

    check_file("./" + folder + "/epoch_%s/" % (epoch))

    for i in range(0, len(gen_imgs)):
        imsave("./" + folder + "/epoch_%s/%s.png" % (epoch, titles[i]), gen_imgs[i])

# TRAINING ############################################################################################################################################

def augmented_data(loaded_images):
    images = np.empty((len(loaded_images), img_height, img_width, channels))
    i = 0
    for single_image in loaded_images:
        if np.random.random() > 0.5:
            single_image = np.fliplr(single_image)

        images[i, :, :, :] = single_image
        i += 1

    shuffled_indexes = np.random.permutation(len(images))
    images = images[shuffled_indexes]

    return images

start_time = datetime.datetime.now()

# Sample a batch of images from both domains
imgs_A = load_data(dataset_A)
imgs_B = load_data(dataset_B)

discriminator_loss = []
generator_loss = []

for epoch in range(epochs):

    imgs_A = augmented_data(imgs_A)
    imgs_B = augmented_data(imgs_B)

    # Translate images to opposite domain
    fake_B = g_AB.predict(imgs_A)
    fake_A = g_BA.predict(imgs_B)

    # The generators want the discriminators to label the translated images as real
    valid = np.ones((batch_size,) + (3, 3, 1))
    fake = np.zeros((batch_size,) + (3, 3, 1))

    n_iterations = min(len(imgs_A), len(imgs_B)) // batch_size
    i = 0
    for it in range (0, n_iterations):
        imgs_A_p = imgs_A[i : i+batch_size]
        fake_A_p = fake_A[i : i+batch_size]
        imgs_B_p = imgs_B[i : i+batch_size]
        fake_B_p = fake_B[i : i+batch_size]

        # TRAIN DISCRIMINATORS 
        dA_loss_real = d_A.train_on_batch(imgs_A_p, valid)    # original images = real
        dA_loss_fake = d_A.train_on_batch(fake_A_p, fake)     # translated images = Fake
        
        dB_loss_real = d_B.train_on_batch(imgs_B_p, valid)
        dB_loss_fake = d_B.train_on_batch(fake_B_p, fake)

        # TRAIN GENERATORS
        g_loss = combined.train_on_batch([imgs_A_p, imgs_B_p], [valid, valid, imgs_A_p, imgs_B_p, imgs_A_p, imgs_B_p])

        i += batch_size

    # Disciminator losses
    dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)
    dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)
    d_loss = 0.5 * np.add(dA_loss, dB_loss)

    elapsed_time = datetime.datetime.now() - start_time
    
    # Plot the progress
    print ("Epoch: %d Time: %s \n D-Loss %s \n G-Loss %s " % (epoch, elapsed_time, d_loss, g_loss))

    # Register the losses
    discriminator_loss.append(d_loss)
    generator_loss.append(g_loss)

    #  Periodical saving
        # If at save interval => save generated image samples
    if epoch % save_interval == 0:
        save_imgs(epoch)

        np.save("./" + folder + "/epoch_%s/d_loss_%s.npy" % (epoch, epoch), discriminator_loss)
        np.save("./" + folder + "/epoch_%s/g_loss_%s.npy" % (epoch, epoch), generator_loss)

        d_A.save_weights("./" + folder + "/epoch_%s/weights_d_A_%s.h5" % (epoch, epoch))
        d_B.save_weights("./" + folder + "/epoch_%s/weights_d_B_%s.h5" % (epoch, epoch))
        g_AB.save_weights("./" + folder + "/epoch_%s/weights_g_AB_%s.h5" % (epoch, epoch))
        g_BA.save_weights("./" + folder + "/epoch_%s/weights_g_BA_%s.h5" % (epoch, epoch))