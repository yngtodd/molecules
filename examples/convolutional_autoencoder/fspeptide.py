import argparse
from keras.optimizers import RMSprop

from molecules.ml.unsupervised import EncoderConvolution2D
from molecules.ml.unsupervised import DecoderConvolution2D
from molecules.ml.unsupervised import VAE

from molecules.data import FSPeptide


def main():
    parser = argparse.ArgumentParser(description='Convolutional VAE for FS-Peptide data.')
    parser.add_argument('--data_path', type=str, help='Path to load fs-peptide data.')
    args = parser.parse_args()

    train_data = FSPeptide(args.data_path, partition='train', download=True)
    val_data = FSPeptide(args.data_path, partition='validation', download=True)

    x_train, _ = train_data.load_data()
    x_val, _ = val_data.load_data()
    input_shape = (21,21,1)

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    encoder = EncoderConvolution2D(input_shape=input_shape)

    #encoder._get_final_conv_params()
    #num_conv_params = encoder.total_conv_params
    #encode_conv_shape = encoder.final_conv_shape

    decoder = DecoderConvolution2D(output_shape=input_shape,
                                   enc_conv_params=7744,
                                   enc_conv_shape=(11,11,64))

    cvae = VAE(input_shape=input_shape,
               latent_dim=3,
               encoder=encoder,
               decoder=decoder,
               optimizer=optimizer)

    cvae.train(x_train, validation_data=x_val, batch_size=512, epochs=10)


if __name__=='__main__':
    main()
