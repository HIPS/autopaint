import pickle
import time
import matplotlib.pyplot as plt

import autograd.numpy.random as npr
import autograd.numpy as np

from autograd import grad

from autopaint.plotting import plot_images
from autopaint.optimizers import adam
from autopaint.neuralnet import make_batches
from autopaint.util import load_mnist
from autopaint.aevb import lower_bound
from autopaint.util import WeightsParser, load_and_pickle_binary_mnist
from autopaint.neuralnet import make_binary_nn,make_gaussian_nn
param_scale = 0.1
samples_per_image = 1
latent_dimensions = 10
hidden_units = 500

def get_pretrained_dec_w():
   with open('parameters.pkl') as f:
        parameters = pickle.load(f)
   params,N_weights_enc,samples_per_image,latent_dimensions,rs = parameters
   dec_w = params[N_weights_enc:len(params)]
   return dec_w

def run_aevb(train_images):
    start_time = time.time()

    # Create aevb function
    # Training parameters

    D = train_images.shape[1]

    enc_layers = [D,hidden_units, hidden_units, 2*latent_dimensions]
    dec_layers = [latent_dimensions,hidden_units, hidden_units, D]

    N_weights_enc, encoder, encoder_log_like = make_gaussian_nn(enc_layers)
    N_weights_dec, decoder, decoder_log_like = make_binary_nn(dec_layers)

    # Optimize aevb
    batch_size = 100
    num_training_iters = 1600
    rs = npr.RandomState(0)


    parser = WeightsParser()
    parser.add_shape('encoding weights', (N_weights_enc,))
    initial_combined_weights = rs.randn(len(parser)) * param_scale
    parser.add_shape('decoding weights', (N_weights_dec,))
    params = np.zeros(len(parser))
    dec_w = get_pretrained_dec_w()
    parser.put(params, 'encoding weights',initial_combined_weights)
    parser.put(params, 'decoding weights',dec_w)




    batch_idxs = make_batches(train_images.shape[0], batch_size)

    def batch_value_and_grad(weights, iter):
        iter = iter % len(batch_idxs)
        cur_data = train_images[batch_idxs[iter]]
        return lower_bound(weights,encoder,decoder_log_like,N_weights_enc,cur_data,samples_per_image,latent_dimensions,rs)
    lb_grad = grad(batch_value_and_grad)

    def callback(params, i, grad):
        ml = batch_value_and_grad(params,i)
        print "log marginal likelihood:", ml

        #Generate samples
        num_samples = 100
        images_per_row = 10
        zs = rs.randn(num_samples,latent_dimensions)
        samples = decoder(parser.get(params, 'decoding weights'), zs)
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
        plot_images(samples, ax, ims_per_row=images_per_row)
        plt.savefig('samples.png')

    final_params = adam(lb_grad, params, num_training_iters, callback=callback)

    def decoder_with_weights(zs):
        return decoder(parser.get(final_params, 'decoding weights'), zs)
    return decoder_with_weights

    finish_time = time.time()
    print "total runtime", finish_time - start_time




if __name__ == '__main__':
    # load_and_pickle_binary_mnist()
    with open('../../../autopaint/mnist_binary_data.pkl') as f:
        N_data, train_images, train_labels, test_images, test_labels = pickle.load(f)

    decoder = run_aevb(train_images)
