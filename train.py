from functools import partial
import numpy as np
import os
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow._api.v2 import random
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dropout
from args import Args
from tensorflow.keras.optimizers import Adam, RMSprop
import discriminator
from generator import build_generator
from discriminator import build_discriminator
from loss import discriminator_loss, generator_loss, wasserstein_loss, gradient_penalty
from data import create_image_path_list, build_dataset, denormalize, generate_image, normalize
import time

def build_gan():

    discriminator = build_discriminator()
    discriminator.summary()

    generator = build_generator()
    generator.summary()

    return discriminator, generator

@tf.function
def train_discriminator(dis, gen, optimizer, real_image):
    dis.trainable = True
    for layer in dis.layers:
        layer.trainable = True
    gen.trainable = False
    for layer in gen.layers:
        layer.trainable = False

    random_vector = tf.random.normal(shape=(Args.batch_size,1,1,Args.noise_shape[2]), seed=int(time.time()))

    with tf.GradientTape() as tape:

        fake_image = gen(random_vector, training=True)

        real_logit = dis(real_image, training=True)
        fake_logit = dis(fake_image, training=True)

        real_loss, fake_loss = discriminator_loss(real_logit, fake_logit)
        gp_loss = gradient_penalty(partial(dis, training=True), real_image, fake_image)

        d_loss = real_loss + fake_loss + gp_loss * Args.gradient_penalty

    gradients = tape.gradient(d_loss, dis.trainable_weights)
    optimizer.apply_gradients(zip(gradients,dis.trainable_weights))
    return real_loss + fake_loss, gp_loss

@tf.function
def train_generator(dis, gen, optimizer):
    gen.trainable = True
    for layer in gen.layers:
        layer.trainable = True
    dis.trainable = False
    for layer in dis.layers:
        layer.trainable = False

    random_vector = tf.random.normal(shape=(Args.batch_size,1,1,Args.noise_shape[2]), seed=int(time.time()))

    with tf.GradientTape() as tape:


        #perform dropout on generator input
        #drop_out = Dropout(0.5, input_shape=random_vector.shape)
        #random_vector = drop_out(random_vector, training=True)

        fake_image = gen(random_vector, training=True)

        fake_logit = dis(fake_image, training=True)

        g_loss = generator_loss(fake_logit)

    gradients = tape.gradient(g_loss, gen.trainable_weights)
    optimizer.apply_gradients(zip(gradients, gen.trainable_weights))

    return g_loss

def run_batch(dis, gen, d_opt, g_opt, train_ds, div):
    for step, image_batch in enumerate(train_ds):
        #train discriminator
        for i in range(Args.dis_gen_train_ratio):
            d_loss, gp_loss = train_discriminator(dis, gen, d_opt, image_batch)

        #saving losses

        #train generator
        #if (step+1)%Args.dis_gen_train_ratio == 0:
        g_loss = train_generator(dis, gen, g_opt)
        if (div + 1 == Args.data_loading_batch_num):
            print("Current d_loss: {}, Current gp_loss: {}, Current g_loss: {}".format(d_loss, gp_loss, g_loss))

            #saving losses
        
def train_gan():
    #save models and model logs
    log_dir = "gan_log"
    g_model_dir = os.path.join(log_dir, 'generator_models')
    d_model_dir = os.path.join(log_dir, 'discriminator_models')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(g_model_dir, exist_ok=True)
    os.makedirs(d_model_dir, exist_ok=True)

    #generate fake images when training
    generated_image_dir = os.path.join(log_dir, 'generated_image_while_training')
    os.makedirs(generated_image_dir, exist_ok=True)

    dis, gen = build_gan()
    image_path_list = create_image_path_list()

    """d_opt = RMSprop(learning_rate=Args.d_learning_rate)
    g_opt = RMSprop(learning_rate=Args.g_learning_rate)"""

    """dis.compile(optimizer=d_opt, loss=discriminator_loss)
    gen.compile(optimizer=g_opt, loss=generator_loss)"""
    d_opt = Adam(learning_rate=Args.d_learning_rate, beta_1=Args.adam_beta, beta_2=0.9)
    g_opt = Adam(learning_rate=Args.g_learning_rate, beta_1=Args.adam_beta, beta_2=0.9)

    for epoch in range(Args.epoch):
        print("Currently at epoch {}.".format(epoch + 1))
        #we build the dataset only when we need the data for training,
        #this can avoid OOM error. The dataset is batched into samller pieces
        #with size Args.data_loading_batch and the total number of image used in
        #training are Args.data_loading_batch * Args.data_loading_batch_nums.

        
        for div in range(Args.data_loading_batch_num):
            print("Training model with {} images in division {}.".
            format(Args.data_loading_batch, div+1))
            train_ds = build_dataset(image_path_list, div)
            run_batch(dis, gen, d_opt, g_opt, train_ds, div)

        if (epoch+1)%1 == 0:
            random_vector = tf.random.normal(shape=(
         100,Args.noise_shape[0],Args.noise_shape[1],Args.noise_shape[2]))
            fake_image = gen(random_vector, training=False)

            """generated_image = np.squeeze(generated_image, axis=0)
            generated_image = denormalize(generated_image)"""

            combined_image = generate_image(fake_image)
            save_dir = os.path.join(generated_image_dir, 
            "generated_image_at_epoch_{}.jpg".format(epoch+1))
            cv2.imwrite(save_dir, cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))

        

if __name__ == "__main__":
    train_gan()
