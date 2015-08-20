import pickle
import time
import matplotlib.pyplot as plt

import autograd.numpy.random as npr
import autograd.numpy as np

from autograd import value_and_grad

from autopaint.plotting import plot_images
from autopaint.optimizers import adam_mini_batch
from autopaint.neuralnet import make_batches
from autopaint.util import load_mnist
from autopaint.aevb import build_encoder, build_binarized_decoder, lower_bound
from autopaint.util import WeightsParser

param_scale = 0.1
samples_per_image = 5
latent_dimensions = 20
hidden_units = 500

def run_aevb(train_images):
    start_time = time.time()

    # Create aevb function
    # Training parameters

    D = train_images.shape[1]

    enc_layers = [D, hidden_units, 2*latent_dimensions]
    dec_layers = [latent_dimensions, hidden_units, D]

    N_weights_enc, encoder = build_encoder(enc_layers)
    N_weights_dec, decoder, decoder_log_like = build_binarized_decoder(dec_layers)

    # Optimize aevb
    batch_size = 256
    num_epochs = 160
    rs = npr.RandomState(0)

    parser = WeightsParser()
    parser.add_shape('encoding weights', (N_weights_enc,))
    parser.add_shape('decoding weights', (N_weights_dec,))
    initial_combined_weights = rs.randn(len(parser)) * param_scale

    batch_idxs = make_batches(train_images.shape[0], batch_size)

    def batch_value_and_grad(weights, idxs):
        def batch_lower_bound(weights):
            return lower_bound(weights,encoder,decoder_log_like,N_weights_enc,train_images[idxs],samples_per_image,latent_dimensions,rs)

        val_and_grad_func = value_and_grad(batch_lower_bound)

        return val_and_grad_func(weights)

    def callback(ml, weights, grad):
        print "log marginal likelihood:", ml

        #Generate samples
        num_samples = 100
        images_per_row = 10
        zs = rs.randn(num_samples,latent_dimensions)
        samples = decoder(parser.get(weights, 'decoding weights'), zs)
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
        plot_images(samples, ax, ims_per_row=images_per_row)
        plt.savefig('samples.png')

    final_params, final_value = adam_mini_batch(batch_value_and_grad, initial_combined_weights,
                                                batch_idxs, num_epochs, callback=callback)

    def decoder_with_weights(zs):
        return decoder(parser.get(final_params, 'decoding weights'), zs)
    return decoder_with_weights

    finish_time = time.time()
    print "total runtime", finish_time - start_time


def load_and_pickle_binary_mnist():
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    train_images = np.round(train_images)
    test_images = np.round(test_images)
    mnist_data = N_data, train_images, train_labels, test_images, test_labels
    with open('mnist_binary_data.pkl', 'w') as f:
        pickle.dump(mnist_data, f, 1)

if __name__ == '__main__':
    #load_and_pickle_binary_mnist()
    with open('mnist_binary_data.pkl') as f:
        N_data, train_images, train_labels, test_images, test_labels = pickle.load(f)

    decoder = run_aevb(train_images)

