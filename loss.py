import tensorflow as tf
import numpy as np
from args import Args



def generator_loss(fake_logit):
    g_loss = -tf.reduce_mean(fake_logit)
    return g_loss

def discriminator_loss(real_logit, fake_logit):
    real_loss = tf.reduce_mean(real_logit)
    fake_loss = -tf.reduce_mean(fake_logit)

    return real_loss,fake_loss

def wasserstein_loss(pred, truth):
    return tf.reduce_mean(pred*truth)

def gradient_penalty(discriminator, real_image, fake_image):
    e = tf.random.normal([Args.batch_size,1,1,1], 0.0, 1.0)
    diff = fake_image - real_image
    inter = real_image + e * diff

    with tf.GradientTape() as tape:
        tape.watch(inter)
        pred = discriminator(inter, training=True)

    gradients = tape.gradient(pred, [inter])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]))
    gp = tf.reduce_mean((norm - 1.0)**2)
    return gp
    