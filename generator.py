from tensorflow.python.keras import activations
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras.layers.core import Activation
from tensorflow.python.ops.gen_math_ops import xlogy_eager_fallback
from args import Args
from tensorflow.keras.layers import Conv2D, UpSampling2D, AveragePooling2D, BatchNormalization, Conv2DTranspose, Dropout, GaussianNoise, Input, LeakyReLU
from tensorflow.keras import models

#Create generator model
def build_generator():
    def _Conv2DTran_Block(l, filter_num, filter_shape=(4,4), strides=(2,2), padding='same'):

        #l = GaussianNoise(stddev=Args.G_noise_stddev)(l)

        """l = Conv2DTranspose(filter_num, filter_shape, padding=padding, 
        strides=strides, kernel_initializer=Args.G_kernal_initializer)(l)"""

        l = UpSampling2D(size=strides)(l)

        l = Conv2D(filter_num, filter_shape, padding='same')(l)

        #l = AveragePooling2D(pool_size = (2,2), padding='same', strides=(1,1))(l)

        l = BatchNormalization(momentum=Args.batch_momentum)(l)

        l = LeakyReLU(alpha=Args.leakyReLU_alpha)(l)

        return l

    #(1,1)
    input = Input(shape=Args.noise_shape)

    x = input

    #(4,4)
    x = _Conv2DTran_Block(x, 256, (4,4), (4,4), 'valid')

    #(8,8)
    x = _Conv2DTran_Block(x, 128)
    
    #(16,16)
    x = _Conv2DTran_Block(x, 64)

    #(32,32)
    x = _Conv2DTran_Block(x, 32)

    #(64,64)
    x = _Conv2DTran_Block(x, 16)

    x = Conv2DTranspose(3, (4,4), padding='same',
    kernel_initializer=Args.G_kernal_initializer)(x)

    x = Activation('tanh')(x)


    return models.Model(inputs=input, outputs=x)



