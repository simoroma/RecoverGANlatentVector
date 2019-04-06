#!/usr/bin/env python3

# This script is a slightly modified version of
# https://gist.github.com/matpalm/23dc5804c6d673b800093d0d15e5de0e
# By Mat Kelcey https://twitter.com/mat_kelcey

# Given a specific latent vector saved in a
# numpy array and a random latent vector
# the GAN generates their corresponding images
# and linearly interpolates the images in
# between them.

# It requires:
# - the starting latent vector ./interpolation_from_start/z_00.npy

import os
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# smooth values from point a to point b.
folder = "./interpolation_from_start/"
STEPS = 50
pt_a = np.load(folder + "z_00.npy")
pt_b = np.random.normal(size=(512))
z = np.empty((STEPS, 512))
for i, alpha in enumerate(np.linspace(start=0.0, stop=1.0, num=STEPS)):
  z[i] = (1 - alpha) * pt_a + alpha * pt_b

# Choose a directory for which you have privileges
# where you download the tfhub model
print('Downloading the model.')
os.environ['TFHUB_CACHE_DIR'] = 'D:/NeuralNetworks/ProGAN'
generator = hub.Module("http://tfhub.dev/google/progan-128/1")
print('Model downloaded.')

# sample all z and write out as separate images.
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  imgs = sess.run(generator(z))
  imgs = (imgs * 255).astype(np.uint8)
  for i, img in enumerate(imgs):
    Image.fromarray(img).save(folder + "foo_%02d.png" % i)