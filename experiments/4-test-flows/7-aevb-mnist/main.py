import pickle
import time
import matplotlib.pyplot as plt

import autograd.numpy.random as npr
import autograd.numpy as np

from autograd import grad
from autograd.util import quick_grad_check

from autopaint.plotting import plot_images
from autopaint.optimizers import adam
from autopaint.neuralnet import make_batches
from autopaint.util import load_mnist, entropy_of_diagonal_gaussians
from autopaint.util import WeightsParser, load_and_pickle_binary_mnist,build_logprob_standard_normal
from autopaint.neuralnet import make_binary_nn,make_gaussian_nn
from autopaint.flows import build_flow_sampler_with_inputs
param_scale = 0.1
samples_per_image = 1
latent_dimensions = 5
hidden_units = 500

def lower_bound(weights,parser,flow_sample,encode,decode_log_like,log_prior,N_weights_enc,train_images,samples_per_image,latent_dimensions,rs):
    mean_log_joint,mean_ent = compute_log_prob_and_ent(weights,parser,flow_sample,encode,decode_log_like,log_prior,train_images,samples_per_image,latent_dimensions,rs)
    print "joint ll average",mean_log_joint
    print "ent average", mean_ent

    return mean_log_joint + mean_ent

def compute_log_prob_and_ent(weights,parser,flow_sample,encode,decode_log_like,log_prior,train_images,samples_per_image,latent_dimensions,rs):
    enc_w = parser.get(weights, 'encoder weights')
    dec_w = parser.get(weights, 'decoder weights')
    (mus,log_sigs) = encode(enc_w,train_images)
    sigs = np.exp(log_sigs)

    noise = rs.randn(samples_per_image,train_images.shape[0],latent_dimensions)
    Z_samples2 = mus + sigs*noise
    Z_samples2 = np.reshape(Z_samples2,(train_images.shape[0]*samples_per_image,latent_dimensions),order = 'F')
    Z_samples, entropy_estimates = flow_sample(weights,mus,log_sigs,samples_per_image,rs)
    mean_ent = np.mean(entropy_estimates)
    train_images_repeat = np.repeat(train_images,samples_per_image,axis=0)
    mean_log_prob = np.mean(decode_log_like(dec_w,Z_samples,train_images_repeat) +log_prior(Z_samples))
    mean_log_prob2 = np.mean(decode_log_like(dec_w,Z_samples2,train_images_repeat) +log_prior(Z_samples2))
    mean_ent2 = np.mean(entropy_of_diagonal_gaussians(sigs))
    return mean_log_prob, mean_ent


def run_aevb(train_images):
    start_time = time.time()

    # Create aevb function
    # Training parameters

    D = train_images.shape[1]

    enc_layers = [D, hidden_units, 2*latent_dimensions]
    dec_layers = [latent_dimensions, hidden_units, D]

    N_weights_enc, encoder, encoder_log_like = make_gaussian_nn(enc_layers)
    N_weights_dec, decoder, decoder_log_like = make_binary_nn(dec_layers)

    # Optimize aevb
    batch_size = 100
    num_training_iters = 1600
    rs = npr.RandomState(0)
    num_steps = 0

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

    log_prior = build_logprob_standard_normal(latent_dimensions)
    def batch_value_and_grad(weights, iter):
        iter = iter % len(batch_idxs)
        cur_data = train_images[batch_idxs[iter]]
        return lower_bound(weights,parser,flow_sample,encoder,decoder_log_like,log_prior,N_weights_enc,cur_data,samples_per_image,latent_dimensions,rs)
    lb_grad = grad(batch_value_and_grad)

    def lb_grad_check(weights):
        return batch_value_and_grad(weights,0)
    quick_grad_check(lb_grad_check,sampler_params)
    print 'checked!'
    kill

    def callback(params, i, grad):
        ml = batch_value_and_grad(params,i)
        print "log marginal likelihood:", ml
        #Generate samples
        num_samples = 100
        images_per_row = 10
        zs = rs.randn(num_samples,latent_dimensions)
        samples = decoder(parser.get(params, 'decoder weights'), zs)
        # samples = np.random.binomial(1,decoder(parser.get(params, 'decoding weights'), zs))

        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
        plot_images(samples, ax, ims_per_row=images_per_row)
        plt.savefig('samples.png')

    final_params = adam(lb_grad, sampler_params, num_training_iters, callback=callback)

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

