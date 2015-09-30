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
from autopaint.util import WeightsParser, load_and_pickle_binary_mnist,build_logprob_mvn
from autopaint.neuralnet import make_binary_nn,make_gaussian_nn
from autopaint.early_stop import build_early_stop
param_scale = 0.1
samples_per_image = 1
latent_dimensions = 2
hidden_units = 500

import autograd.numpy as np
from autopaint.util import neg_kl_diag_normal

def lower_bound(weights,encode,decode_log_like,N_weights_enc,train_images,samples_per_image,latent_dimensions,rs):
    enc_w = weights[0:N_weights_enc]
    dec_w = weights[N_weights_enc:len(weights)]
    log_normal = build_logprob_mvn(np.zeros(latent_dimensions), np.eye(latent_dimensions),pseudo_inv = True)
    #Choose an image from train_images
    for idx in xrange(train_images.shape[0]):
        x = train_images[idx,:]
        def log_lik_func(z):
            return decode_log_like(dec_w,z,x) +log_normal(z)
        sample, loglik_estimate, entropy_estimate = encode(enc_w,log_lik_func,rs,1)
        if idx == 0:
            samples = sample
            loglik_estimates = loglik_estimate
            entropy_estimates = entropy_estimate
        else:
            samples = np.concatenate((samples,sample),axis = 0)
            loglik_estimates = np.concatenate((loglik_estimates, loglik_estimate),axis = 0)
            entropy_estimates = np.concatenate((entropy_estimates, entropy_estimate),axis = 0)
    print "ll average", loglik_estimate
    print "kl average", entropy_estimate
    return loglik_estimate + entropy_estimate



def run_aevb(train_images):
    start_time = time.time()

    # Create aevb function
    # Training parameters

    D = train_images.shape[1]

    dec_layers = [latent_dimensions, hidden_units, D]

    init_mean = np.zeros(latent_dimensions)
    init_log_stddevs = np.log(1.0*np.ones(latent_dimensions))
    init_log_stepsize = np.log(0.01)

    rs = np.random.npr.RandomState(0)
    sample_and_run_early_stop, parser = build_early_stop(latent_dimensions, approx=True)
    N_weights_dec, decoder, decoder_log_like = make_binary_nn(dec_layers)
    N_weights_enc = len(parser)
    encoder = sample_and_run_early_stop
    parser.add_shape('decoding weights', (N_weights_dec,))

    params = np.zeros(len(parser))
    parser.put(params, 'mean', init_mean)
    parser.put(params, 'log_stddev', init_log_stddevs)
    parser.put(params, 'log_stepsize', init_log_stepsize)
    parser.put(params, 'decoding weights',rs.randn(N_weights_dec) * param_scale)

    # Optimize aevb
    batch_size = 2
    num_training_iters = 1600
    rs = npr.RandomState(0)

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
        # samples = np.random.binomial(1,decoder(parser.get(params, 'decoding weights'), zs))
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

