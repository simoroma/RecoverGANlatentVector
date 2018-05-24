#!/usr/bin/env python3

# This script is a (super) slightly modified version of
# https://gist.github.com/matpalm/23dc5804c6d673b800093d0d15e5de0e
# By Mat Kelcey https://twitter.com/mat_kelcey

# Given two random latent vectors the GAN generates their
# corresponding images and linearly interpolates the images in
# between them

from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# smooth values from point a to point b.
folder = "./interpolation/"
STEPS = 50
pt_a = np.random.normal(size=(512))
pt_b = np.random.normal(size=(512))
z = np.empty((STEPS, 512))
for i, alpha in enumerate(np.linspace(start=0.0, stop=1.0, num=STEPS)):
  z[i] = (1-alpha) * pt_a + alpha * pt_b

# sample all z and write out as separate images.
generator = hub.Module("https://tfhub.dev/google/progan-128/1")
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  imgs = sess.run(generator(z))
  imgs = (imgs * 255).astype(np.uint8)
  for i, img in enumerate(imgs):
    Image.fromarray(img).save(folder + "foo_%02d.png" % i)
    # save the latent vectors that generated the images
    np.save(folder + "z_%02d" % i, z[i])