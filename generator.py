from tensorflow.python.keras import activations
from tensorflow.python.keras.layers.core import Activation
from args import Args
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2DTranspose, Conv2D, Dropout, Dense, AveragePooling2D, GaussianNoise, Input, LeakyReLU, UpSampling2D, Reshape
from tensorflow.keras import models
from tensorflow.keras.layers.experimental.preprocessing import Resizing

#Create generator model
def build_generator():
    def _Conv2DTran_Block(l, input_height, input_width, filter_num, filter_shape=(4,4), strides=(2,2), padding='same', mutiplier=2):

        l = GaussianNoise(stddev=Args.G_noise_stddev)(l)

        l = Conv2DTranspose(filter_num, filter_shape, padding=padding, 
        strides=strides, kernel_initializer=Args.G_kernal_initializer)(l)

        """l = UpSampling2D(size=(mutiplier, mutiplier))(l)
        l = Conv2D(filter_num, filter_shape, strides=(1,1), padding='same', use_bias=False)(l)"""

        l = AveragePooling2D(pool_size=(2,2), strides=(1,1), padding='same')(l)

        l = BatchNormalization(momentum=Args.batch_momentum)(l)

        l = LeakyReLU(alpha=Args.leakyReLU_alpha)(l)

        #l = Activation(activations.tanh)(l)

        return l

    #(1,1)
    input = Input(shape=Args.noise_shape)

    #(4,4)
    x = Dense(4*4*512, use_bias=False)(input)
    x = Reshape((4, 4, 512))(x)

    #(8,8)
    x = _Conv2DTran_Block(x, 4, 4, 256)
    
    #(16,16)
    x = _Conv2DTran_Block(x, 8, 8, 128)

    #(32,32)
    x = _Conv2DTran_Block(x, 16, 16, 64)

    #(64,64)
    x = _Conv2DTran_Block(x, 32, 32, 32)

    x = Conv2D(3, (4,4), padding='same', activation='tanh',
    kernel_initializer=Args.G_kernal_initializer)(x)

 

    return models.Model(inputs=input, outputs=x)



