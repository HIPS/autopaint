# Main demo script
import time
import pickle

import matplotlib.pyplot as plt

import autograd.numpy as np
from autograd import grad

from autopaint.flows import build_flow_sampler_with_inputs
from autopaint.neuralnet import make_batches, make_gaussian_nn,make_binary_nn
from autopaint.util import sample_from_normal_bimodal, load_and_pickle_binary_mnist
from autopaint.plotting import plot_images
from autopaint.optimizers import adam, adadelta,adagrad

#AEVB params
param_scale = 0.1
samples_per_image = 1
latent_dimensions = 40
hidden_units = 100

if __name__ == '__main__':

    start_time = time.time()
    rs = np.random.npr.RandomState(0)
    #load_and_pickle_binary_mnist()
    with open('../../../autopaint/mnist_binary_data.pkl') as f:
        N_data, train_images, train_labels, test_images, test_labels = pickle.load(f)

    #Create aevb function
    D = train_images.shape[1]

    enc_layers = [D, hidden_units, 2*latent_dimensions]
    dec_layers = [latent_dimensions, hidden_units, D]

    N_weights_enc, encoder, encoder_log_like = make_gaussian_nn(enc_layers)
    N_weights_dec, decoder, decoder_log_like = make_binary_nn(dec_layers)

    #Optimize aevb
    batch_size = 100
    num_train_iters = 1000
    num_steps = 1
    num_sampler_optimization_steps = 400
    sampler_learn_rate = 0.01

    init_enc_w = rs.randn(N_weights_enc) * param_scale
    init_dec_w = rs.randn(N_weights_dec) * param_scale
    init_output_weights = 0.1*rs.randn(num_steps, latent_dimensions)
    init_transform_weights = 0.1*rs.randn(num_steps, latent_dimensions)
    init_biases = 0.1*rs.randn(num_steps)

    flow_sample, parser = build_flow_sampler_with_inputs(latent_dimensions, num_steps)
    parser.add_shape('encoder weights',len(init_enc_w))
    parser.add_shape('decoder weights',len(init_dec_w))

    sampler_params = np.zeros(len(parser))
    parser.put(sampler_params, 'output weights', init_output_weights)
    parser.put(sampler_params, 'transform weights', init_transform_weights)
    parser.put(sampler_params, 'biases', init_biases)
    parser.put(sampler_params,'encoder weights', init_enc_w)
    parser.put(sampler_params,'decoder weights', init_dec_w)
    batch_idxs = make_batches(train_images.shape[0], batch_size)

    def get_batch_lower_bound(cur_sampler_params, iter):
        iter = iter % len(batch_idxs)
        cur_data = train_images[batch_idxs[iter]]
        #Create an initial encoding of sample images:
        enc_w = parser.get(cur_sampler_params, 'encoder weights')
        (mus,log_sigs) = encoder(enc_w,cur_data)
        #Take mean of encodings and sigs and use this to generate samples, should return only entropy
        samples, entropy_estimates = flow_sample(cur_sampler_params,mus,log_sigs,samples_per_image,rs)
        #From samples decode them and compute likelihood
        dec_w = parser.get(cur_sampler_params, 'decoder weights')
        train_images_repeat = np.repeat(cur_data,samples_per_image,axis=0)
        loglike = np.mean(decoder_log_like(dec_w,samples,train_images_repeat))
        print "Mean loglik:", loglike,\
        "Mean entropy:", np.mean(entropy_estimates)
        return np.mean(entropy_estimates)+loglike

    def callback(params, i, grad):
        ml = get_batch_lower_bound(params,i)
        print "log marginal likelihood:", ml

        #Generate samples
        num_samples = 100
        images_per_row = 10
        zs = rs.randn(num_samples,latent_dimensions)
        samples = decoder(parser.get(params, 'decoder weights'), zs)
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
        plot_images(samples, ax, ims_per_row=images_per_row)
        plt.savefig('samples.png')

    lb_grad = grad(get_batch_lower_bound)

    final_params = adam(lb_grad, sampler_params,
                                                num_train_iters, callback=callback,alpha =  .01)

    finish_time = time.time()
    print "total runtime", finish_time - start_time


