import os
import argparse
import numpy as np

from keras.optimizers import RMSprop

from molecules.ml.unsupervised import VAE
from molecules.ml.unsupervised import EncoderConvolution2D
from molecules.ml.unsupervised import DecoderConvolution2D
from molecules.ml.unsupervised.callbacks import EmbeddingCallback

from molecules.data import FSPeptide


def main():
    parser = argparse.ArgumentParser(description='Convolutional VAE for FS-Peptide data.')
    parser.add_argument('--data_path', type=str, help='Path to load fs-peptide data.')
    parser.add_argument('--weight_path', type=str, help='Path to save network weights.')
    parser.add_argument('--embedding_path', type=str, help='Path to save embeddings.')

    args = parser.parse_args()

    train_data = FSPeptide(args.data_path, partition='train', download=True)
    val_data = FSPeptide(args.data_path, partition='validation', download=True)

    x_train, _ = train_data.load_data()
    x_val, _ = val_data.load_data()

    # Keras complains if height and width dimensions are odd.
    x_train = np.pad(x_train, [(0,0),(0,1), (0,1), (0,0)], mode='constant')
    x_val = np.pad(x_val, [(0,0),(0,1), (0,1), (0,0)], mode='constant')
    input_shape = (22,22,1)

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

    callback = EmbeddingCallback(x_train, cvae)
    cvae.train(x_train, validation_data=x_val, batch_size=512, epochs=100, callbacks=[callback])

    weight_path = os.path.join(args.weight_path, 'cvae_fspeptide.h5')
    cvae.save_weights(weight_path)
    callback.save_embeddings(filename='fspeptide', path=args.embedding_path)


if __name__=='__main__':
    main()
