class Args:

    epoch = 500000

    image_size = 64

    data_loading_batch = 400

    data_loading_batch_num = 20

    batch_size = 40

    dis_gen_train_ratio = 1

    d_learning_rate = 0.0001

    g_learning_rate = 0.0003

    clipping_rate = 0.01

    gradient_penalty = 10

    d_real_to_fake_loss_ratio = 0.5

    noise_shape = (1,1,128)

    D_kernal_initializer = 'glorot_uniform'

    G_kernal_initializer = 'glorot_uniform'

    D_noise_stddev = 0.1

    G_noise_stddev = 0.1

    image_dir = 'image'

    drop_out_rate = 0.2

    label_noise = 0.1

    batch_momentum = 0.2

    leakyReLU_alpha = 0.5

    adam_beta = 0

