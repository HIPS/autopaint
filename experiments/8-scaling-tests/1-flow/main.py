# Compares various recogntion networks on the same generative model.
import time
import pickle
import matplotlib.pyplot as plt

import autograd.numpy as np
from autograd import grad,value_and_grad

from autopaint.flows import build_flow_sampler
from autopaint.neuralnet import make_batches, make_gaussian_nn, make_binary_nn
from autopaint.optimizers import adam
from autopaint.util import load_and_pickle_binary_mnist, WeightsParser
from autopaint.plotting import plot_images, plot_line

# Model parameters.
param_scale = 0.1
hidden_units = 500
num_flow_steps = 32

def init_flow_params(flow_parser, rs,latent_dimension):
    init_output_weights =    0.1*rs.randn(num_flow_steps, latent_dimension)
    init_transform_weights = 0.1*rs.randn(num_flow_steps, latent_dimension)
    init_biases =            0.1*rs.randn(num_flow_steps)

    params = np.zeros(len(flow_parser))
    flow_parser.put(params, 'output weights', init_output_weights)
    flow_parser.put(params, 'transform weights', init_transform_weights)
    flow_parser.put(params, 'biases', init_biases)
    return params


def time_and_acc(latent_dimension):

    start_time = time.time()
    rs = np.random.npr.RandomState(0)
    #load_and_pickle_binary_mnist()
    with open('../../../autopaint/mnist_binary_data.pkl') as f:
        N_data, train_images, train_labels, test_images, test_labels = pickle.load(f)

    D = train_images.shape[1]
    enc_layer_sizes = [D, hidden_units, 2 * latent_dimension]
    dec_layer_sizes = [latent_dimension, hidden_units, D]

    N_weights_enc, encoder, encoder_log_like = make_gaussian_nn(enc_layer_sizes)
    N_weights_dec, decoder, decoder_log_like = make_binary_nn(dec_layer_sizes)

    # Optimization parameters.
    batch_size = 100
    num_training_iters = 100
    sampler_learn_rate = 0.01
    batch_idxs = make_batches(train_images.shape[0], batch_size)

    init_enc_w = rs.randn(N_weights_enc) * param_scale
    init_dec_w = rs.randn(N_weights_dec) * param_scale

    flow_sampler, flow_parser = build_flow_sampler(latent_dimension, num_flow_steps)

    combined_parser = WeightsParser()
    combined_parser.add_shape('encoder weights', N_weights_enc)
    combined_parser.add_shape('decoder weights', N_weights_dec)
    combined_parser.add_shape('flow params', len(flow_parser))

    combined_params = np.zeros(len(combined_parser))
    combined_parser.put(combined_params, 'encoder weights', init_enc_w)
    combined_parser.put(combined_params, 'flow params', init_flow_params(flow_parser, rs,latent_dimension))
    combined_parser.put(combined_params, 'decoder weights', init_dec_w)



    def get_batch_lower_bound(cur_params, iter):
        encoder_weights = combined_parser.get(cur_params, 'encoder weights')
        flow_params     = combined_parser.get(cur_params, 'flow params')
        decoder_weights = combined_parser.get(cur_params, 'decoder weights')

        cur_data = train_images[batch_idxs[iter]]
        mus, log_sigs = encoder(encoder_weights, cur_data)
        samples, entropy_estimates = flow_sampler(flow_params, mus, np.exp(log_sigs), rs)
        loglikes = decoder_log_like(decoder_weights, samples, cur_data)

        print "Iter", iter, "loglik:", np.mean(loglikes).value, \
            "entropy:", np.mean(entropy_estimates).value, "marg. like:", np.mean(entropy_estimates + loglikes).value
        return np.mean(entropy_estimates + loglikes)

    lb_grad = grad(get_batch_lower_bound)

    def callback(weights, iter, grad):
        #Generate samples
        num_samples = 100
        zs = rs.randn(num_samples, latent_dimension)
        samples = decoder(combined_parser.get(weights, 'decoder weights'), zs)
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
        plot_images(samples, ax, ims_per_row=10)
        plt.savefig('samples.png')


    final_params = adam(lb_grad, combined_params, num_training_iters, callback=callback)

    finish_time = time.time()
    #Broken and very mysterious:
    lb_val_grad = value_and_grad(get_batch_lower_bound)
    lb_est = lb_val_grad(final_params,num_training_iters+2)
    print lb_est
    lb_est = lb_est[0]
    print "Total training time:", finish_time - start_time
    return finish_time, lb_est

if __name__ == '__main__':
    # latent_dimensions = [2,4,8,16,32,64,128]
    latent_dimensions = [32,64,128]
    time_list = []
    bound_list = []
    for d in latent_dimensions:
        t,bound = time_and_acc(d)
        time_list.append(t)
        bound_list.append(bound)
    plot_line(latent_dimensions,time_list,'time_scale_NF.png')
    plot_line(latent_dimensions,bound_list,'bound_scale_NF.png')


