# def log_lik_func_single_sample(z,i):
#     # train_images_repeat = np.repeat(train_images,samples_per_image,axis=0)
#     assert z.shape[0] == 1
#     return decode_log_like(dec_w,z,train_images[i,:])+log_normal(z)
# single_grad = grad(log_lik_func_single_sample)
# ele_grad = elementwise_grad(log_lik_func)
# print single_grad(1.0*np.zeros((1,2)),0) - ele_grad(1.0*np.zeros((100,2)))[0,:]
# kill

import sys
import os

import pickle
import time
import matplotlib.pyplot as plt

import autograd.numpy.random as npr
import autograd.numpy as np

from autograd import grad, elementwise_grad
from autograd.util import quick_grad_check


from autopaint.plotting import plot_images
from autopaint.optimizers import adam
from autopaint.neuralnet import make_batches
from autopaint.util import load_mnist
from autopaint.aevb import lower_bound
from autopaint.util import WeightsParser, load_and_pickle_binary_mnist,build_logprob_mvn
from autopaint.neuralnet import make_binary_nn,make_gaussian_nn
from autopaint.early_stop import build_early_stop_fixed_params
param_scale = 0.1
samples_per_image = 10
latent_dimensions = 10
hidden_units = 500

def get_pretrained_dec_w():
   with open('parameters.pkl') as f:
        parameters = pickle.load(f)
   params,N_weights_enc,samples_per_image,latent_dimensions,rs = parameters
   dec_w = params[N_weights_enc:len(params)]
   return dec_w




import autograd.numpy as np
from autopaint.util import neg_kl_diag_normal


def lower_bound(weights,encode,decode_log_like,N_weights_enc,train_images,samples_per_image,latent_dimensions,rs):
    train_images = np.atleast_2d(train_images)
    enc_w = weights[0:N_weights_enc]
    dec_w = weights[N_weights_enc:len(weights)]
    log_normal = build_logprob_mvn(np.zeros(latent_dimensions), np.eye(latent_dimensions),pseudo_inv = True)

    def log_lik_func(z):
        # train_images_repeat = np.repeat(train_images,samples_per_image,axis=0)
        train_images_repeat = np.repeat(train_images,z.shape[0],axis=0)
        assert z.shape[0] == train_images_repeat.shape[0]
        return decode_log_like(dec_w,z,train_images_repeat) +log_normal(z)


    samples, loglik_estimates, entropy_estimates = encode(enc_w,log_lik_func,rs,num_samples=samples_per_image)
    loglik_estimate = np.mean(loglik_estimates)
    entropy_estimate = np.mean(entropy_estimates)
    # print "ll average", loglik_estimate
    # print "ent average", entropy_estimate
    return loglik_estimate + entropy_estimate


def lower_bound_batch(weights,encode,decode_log_like,N_weights_enc,train_images,samples_per_image,latent_dimensions,rs):
    lower_bound_sum = 0.0
    #Turn off printing
    # sys.stdout = open(os.devnull, "w")

    #Batch_size is train_images.shape[0]
    for i in xrange(train_images.shape[0]):
        cur_image = train_images[i,:]
        lower_bound_sum = lower_bound_sum+lower_bound(weights,encode,decode_log_like,N_weights_enc,cur_image,samples_per_image,latent_dimensions,rs)
    lower_bound_est = lower_bound_sum/train_images.shape[0]
    #Turn on printing
    # sys.stdout = sys.__stdout__

    return lower_bound_est


def run_aevb(train_images):

    # run_aevb(train_images)


    start_time = time.time()

    # Create aevb function
    # Training parameters

    D = train_images.shape[1]

    dec_layers = [latent_dimensions, hidden_units,hidden_units, D]

    mean = np.zeros(latent_dimensions)
    log_stddevs = np.log(1.0*np.ones(latent_dimensions))
    log_stepsize = np.log(.005)

    rs = np.random.npr.RandomState(0)
    sample_and_run_es =build_early_stop_fixed_params( latent_dimensions,  approx=True,mean= mean,log_stddevs = log_stddevs,log_stepsize=log_stepsize)
    N_weights_dec, decoder, decoder_log_like = make_binary_nn(dec_layers)
    N_weights_enc = 0
    encoder = sample_and_run_es
    # Build parser
    parser = WeightsParser()
    parser.add_shape('decoding weights', (N_weights_dec,))

    params = np.zeros(len(parser))
    parser.put(params, 'decoding weights',rs.randn(N_weights_dec) * param_scale)
    assert len(parser) == N_weights_dec

    # Optimize aevb
    batch_size = 10
    num_training_iters = 1600
    rs = npr.RandomState(0)

    batch_idxs = make_batches(train_images.shape[0], batch_size)

    def batch_value_and_grad(weights, iter):
        iter = iter % len(batch_idxs)
        cur_data = train_images[batch_idxs[iter]]
        return lower_bound_batch(weights,encoder,decoder_log_like,N_weights_enc,cur_data,samples_per_image,latent_dimensions,rs)
    lb_grad = grad(batch_value_and_grad)


    def callback(params, i, grad):
        n_iter= 0.0
        sum_ml = 0
        for j in xrange(0,1):
            ml = batch_value_and_grad(params,j)
            print "---- log marginal likelihood:", ml
            n_iter += 1
            sum_ml += ml
            print '-------- avg_ml', sum_ml/n_iter


        #Generate samples
        num_samples = 10
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

