from tensorflow.python.keras import activations
from args import Args
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, GaussianNoise, Flatten, Input, LeakyReLU
from tensorflow.keras import models
from tensorflow.keras.constraints import max_norm

#create discriminator
def build_discriminator():
    def _Conv2D_Block(l, filter_num, filter_shape=(4,4), strides=(2,2), padding='same'):

        l = Conv2D(filter_num, filter_shape, 
        strides=strides, padding=padding, kernel_initializer=Args.D_kernal_initializer, use_bias=False)(l)

        l = BatchNormalization(momentum=Args.batch_momentum)(l)

        l = LeakyReLU(alpha=Args.leakyReLU_alpha)(l)

        l = Dropout(Args.drop_out_rate)(l)

        #l = Activation(activations.tanh)(l)

        return l
    
    #(64,64)
    input = Input(shape=(Args.image_size, Args.image_size,3))

    x = input

    x = GaussianNoise(stddev=Args.D_noise_stddev)(x)

    #(32,32)
    x = Conv2D(32, (3,3), strides=(2,2), padding='same', kernel_initializer=Args.D_kernal_initializer)(x)
    x = LeakyReLU(alpha=Args.leakyReLU_alpha)(x)
    x = Dropout(Args.drop_out_rate)(x)

    #(16,16)
    x = _Conv2D_Block(x, 64)

    #(8,8)
    x = _Conv2D_Block(x, 128)

    #(4,4)
    x = _Conv2D_Block(x, 256)

    #(2,2)
    x = _Conv2D_Block(x , 512)

    x = Flatten()(x)

    #(1,1)
    #I did not use a sigmoid as activation since we don't use cross-entropy as loss function
    x = Dense(1, kernel_initializer=Args.D_kernal_initializer)(x)
    x = Dropout(Args.drop_out_rate)(x)

    return models.Model(inputs=input, outputs=x)

