# Compares various recogntion networks on the same generative model.
import time
import pickle
import matplotlib.pyplot as plt

import autograd.numpy as np
from autograd import value_and_grad

from autopaint.flows import build_flow_sampler
from autopaint.neuralnet import make_batches
from autopaint.aevb import build_encoder, build_binarized_decoder
from autopaint.optimizers import adam_mini_batch
from autopaint.util import load_and_pickle_binary_mnist, WeightsParser
from autopaint.plotting import plot_images

#AEVB params
param_scale = 0.1
latent_dimension = 2
hidden_units = 500

def init_flow_params(flow_parser, rs):
    init_output_weights = 0.1*rs.randn(num_steps, latent_dimension)
    init_transform_weights = 0.1*rs.randn(num_steps, latent_dimension)
    init_biases = 0.1*rs.randn(num_steps)

    params = np.zeros(len(flow_parser))
    flow_parser.put(params, 'output weights', init_output_weights)
    flow_parser.put(params, 'transform weights', init_transform_weights)
    flow_parser.put(params, 'biases', init_biases)

    return params


if __name__ == '__main__':

    start_time = time.time()
    rs = np.random.npr.RandomState(0)
    #load_and_pickle_binary_mnist()
    with open('mnist_binary_data.pkl') as f:
        N_data, train_images, train_labels, test_images, test_labels = pickle.load(f)

    D = train_images.shape[1]
    enc_layer_sizes = [D, hidden_units, 2 * latent_dimension]
    dec_layer_sizes = [latent_dimension, hidden_units, D]

    N_weights_enc, encoder = build_encoder(enc_layer_sizes)
    N_weights_dec, decoder, decoder_log_like = build_binarized_decoder(dec_layer_sizes)

    # Optimize aevb
    batch_size = 100
    num_epochs = 1
    num_steps = 32
    num_sampler_optimization_steps = 400
    sampler_learn_rate = 0.01

    init_enc_w = rs.randn(N_weights_enc) * param_scale
    init_dec_w = rs.randn(N_weights_dec) * param_scale

    flow_sampler, flow_parser = build_flow_sampler(latent_dimension, num_steps)

    combined_parser = WeightsParser()
    combined_parser.add_shape('encoder weights', N_weights_enc)
    combined_parser.add_shape('decoder weights', N_weights_dec)
    combined_parser.add_shape('flow params', len(flow_parser))

    combined_params = np.zeros(len(combined_parser))
    combined_parser.put(combined_params, 'encoder weights', init_enc_w)
    combined_parser.put(combined_params, 'flow params', init_flow_params(flow_parser, rs))
    combined_parser.put(combined_params, 'decoder weights', init_dec_w)
    batch_idxs = make_batches(train_images.shape[0], batch_size)

    def get_batch_lower_bound(cur_sampler_params, idxs):
        #Create an initial encoding of sample images:
        encoder_weights = combined_parser.get(cur_sampler_params, 'encoder weights')
        flow_params     = combined_parser.get(cur_sampler_params, 'flow params')
        decoder_weights = combined_parser.get(cur_sampler_params, 'decoder weights')

        mus, log_sigs = encoder(encoder_weights, train_images[idxs])
        samples, entropy_estimates = flow_sampler(flow_params, mus, np.exp(log_sigs), rs)
        loglike = decoder_log_like(decoder_weights, samples, train_images[idxs])

        print "Mean loglik:", loglike.value, "Mean entropy:", np.mean(entropy_estimates.value)
        return np.mean(entropy_estimates) + loglike

    lb_val_grad = value_and_grad(get_batch_lower_bound)

    def callback(ml, weights, grad):
        print "log marginal likelihood:", ml

        #Generate samples
        num_samples = 100
        images_per_row = 10
        zs = rs.randn(num_samples, latent_dimension)
        samples = decoder(combined_parser.get(weights, 'decoder weights'), zs)
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
        plot_images(samples, ax, ims_per_row=images_per_row)
        plt.savefig('samples.png')

    final_params, final_value = adam_mini_batch(lb_val_grad, combined_params,
                                                batch_idxs, num_epochs, callback=callback)

    finish_time = time.time()
    print "Total training time:", finish_time - start_time


