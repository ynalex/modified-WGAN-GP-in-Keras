from tensorflow.python.keras import activations
from tensorflow.python.keras.constraints import MinMaxNorm
from args import Args
from tensorflow.keras.layers import AveragePooling2D,LayerNormalization, BatchNormalization, Conv2D, Dense, Dropout, GaussianNoise, Flatten, Input, LeakyReLU
from tensorflow.keras import models
from tensorflow.keras.constraints import max_norm

#create discriminator
def build_discriminator():
    def _Conv2D_Block(l, filter_num, filter_shape=(4,4), strides=(2,2), padding='same'):

        l = Conv2D(filter_num, filter_shape, 
        strides=strides, padding=padding, kernel_initializer=Args.D_kernal_initializer, kernel_constraint=MinMaxNorm(-0.1,0.1))(l)

        #l = BatchNormalization(momentum=Args.batch_momentum)(l)

        #l = LayerNormalization()(l)

        l = LeakyReLU(alpha=Args.leakyReLU_alpha)(l)

        #l = AveragePooling2D(pool_size = (2,2), padding='same', strides=(1,1))(l)

        #l = Dropout(Args.drop_out_rate)(l)

        return l
    
    #(64,64)
    input = Input(shape=(Args.image_size, Args.image_size,3))

    x = input

    #(32,32)
    x = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=Args.D_kernal_initializer, kernel_constraint=MinMaxNorm(-0.1,0.1))(x)
    x = LeakyReLU(alpha=Args.leakyReLU_alpha)(x)
    #x = Dropout(Args.drop_out_rate)(x)
    #(16,16)
    x = _Conv2D_Block(x, 128)

    #(8,8)
    x = _Conv2D_Block(x, 256)

    #(4,4)
    x = _Conv2D_Block(x, 512)


    x = Flatten()(x)

    #(1,1)
    #I did not use a sigmoid as activation since we don't use cross-entropy as loss function
    x = Dense(1, kernel_initializer=Args.D_kernal_initializer)(x)

    return models.Model(inputs=input, outputs=x)

