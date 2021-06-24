import tensorflow as tf
import numpy as np
from args import Args



def generator_loss(fake_logit):
    g_loss = -tf.reduce_mean(fake_logit)
    return g_loss

def discriminator_loss(real_logit, fake_logit):
    real_loss = -tf.reduce_mean(real_logit)
    fake_loss = tf.reduce_mean(fake_logit)

    return real_loss, fake_loss

def wasserstein_loss(pred, truth):
    return tf.reduce_mean(pred*truth)

def gradient_penalty(discriminator, real_img, fake_img):
    def interpolate(x, y):
        #shape = [tf.shape(x)[0]] + [1] * (x.shape.ndims - 1)
        alpha = tf.random.normal([Args.batch_size, 1, 1, 1], 0.0, 1.0)
        inter = (alpha * x) + (1 - alpha) * y
        return inter 
    x = interpolate(real_img, fake_img)
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred_logit = discriminator(x)
    grads = tape.gradient(pred_logit, x)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)    
    return gp