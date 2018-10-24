from keras.datasets import mnist
from keras.optimizers import RMSprop

from molecules.ml.unsupervised import EncoderConvolution2D
from molecules.ml.unsupervised import DecoderConvolution2D
from molecules.ml.unsupervised import VAE


def main():
    img_rows = img_cols = 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    encoder = EncoderConvolution2D(input_shape=input_shape)

    encoder._get_final_conv_params()
    num_conv_params = encoder.total_conv_params
    encode_conv_shape = encoder.final_conv_shape

    decoder = DecoderConvolution2D(output_shape=input_shape,
                                   enc_conv_params=num_conv_params,
                                   enc_conv_shape=encode_conv_shape)

    cvae = VAE(input_shape=input_shape,
               latent_dim=3,
               encoder=encoder,
               decoder=decoder,
               optimizer=optimizer)

    cvae.train(x_train, validation_data=x_test, batch_size=10, epochs=10)


if __name__=='__main__':
    main()
