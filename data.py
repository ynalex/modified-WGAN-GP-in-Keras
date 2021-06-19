import re
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow.python.ops.gen_math_ops import imag
from args import Args
import numpy as np
import tensorflow as tf
import os
AUTOTUNE = tf.data.experimental.AUTOTUNE

#map image from 0~255 to -1~1
def normalize(image):
    image = tf.cast(image, tf.float32)
    image = image/127.5 - 1
    return image

#map image from -1~1 to 0~255
def denormalize(image):
    image = tf.cast(image, tf.float32)
    image = ((image + 1) / 2) * 255
    return image

#create a list that contain path of images for training
def create_image_path_list():
    image_path_list = []
    image_count = 0
    for image in sorted(os.listdir(Args.image_dir)):
        if image_count == int(Args.data_loading_batch*Args.data_loading_batch_num):
            break
        image_path_list.append(os.path.join(Args.image_dir,image))
        image_count += 1
    return image_path_list
    
#build tf.data.Dataset
def build_dataset(image_path_list, div):
    #load image from image path and normalize it
    def _load_and_preprocessing(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image)
        image = normalize(image)
        return image

    image_path_list_batched = image_path_list[(div)*Args.data_loading_batch:(div+1)*Args.data_loading_batch]
    path_ds = tf.data.Dataset.from_tensor_slices(image_path_list_batched)
    image_ds = path_ds.map(_load_and_preprocessing)
    image_ds = image_ds.shuffle(buffer_size=Args.data_loading_batch)
    image_ds = image_ds.batch(Args.batch_size)
    image_ds = image_ds.prefetch(buffer_size=AUTOTUNE)

    return image_ds

