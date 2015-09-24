# def log_lik_func_single_sample(z,i):
#     # train_images_repeat = np.repeat(train_images,samples_per_image,axis=0)
#     assert z.shape[0] == 1
#     return decode_log_like(dec_w,z,train_images[i,:])+log_normal(z)
# single_grad = grad(log_lik_func_single_sample)
# ele_grad = elementwise_grad(log_lik_func)
# print single_grad(1.0*np.zeros((1,2)),0) - ele_grad(1.0*np.zeros((100,2)))[0,:]
# kill


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
from autopaint.early_stop import build_early_stop
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
    print "ll average", loglik_estimate
    print "ent average", entropy_estimate
    return loglik_estimate + entropy_estimate



def run_aevb(train_images):

    # run_aevb(train_images)


    start_time = time.time()

    # Create aevb function
    # Training parameters

    D = train_images.shape[1]

    dec_layers = [latent_dimensions, hidden_units,hidden_units, D]

    init_mean = np.zeros(latent_dimensions)
    init_log_stddevs = np.log(1.0*np.ones(latent_dimensions))
    init_log_stepsize = np.log(.001)

    rs = np.random.npr.RandomState(0)
    sample_and_run_es, parser =build_early_stop( latent_dimensions,  approx=True)
    N_weights_dec, decoder, decoder_log_like = make_binary_nn(dec_layers)
    N_weights_enc = len(parser)
    encoder = sample_and_run_es
    parser.add_shape('decoding weights', (N_weights_dec,))

    params = np.zeros(len(parser))
    parser.put(params, 'mean', init_mean)
    parser.put(params, 'log_stddev', init_log_stddevs)
    parser.put(params, 'log_stepsize', init_log_stepsize)
    dec_w = get_pretrained_dec_w()
    parser.put(params, 'decoding weights',dec_w)
    # parser.put(params, 'decoding weights',rs.randn(N_weights_dec) * param_scale)

    # Optimize aevb
    batch_size = 1
    num_training_iters = 1600
    rs = npr.RandomState(0)

    batch_idxs = make_batches(train_images.shape[0], batch_size)

    def batch_value_and_grad(weights, iter):
        iter = iter % len(batch_idxs)
        cur_data = train_images[batch_idxs[iter]]
        return lower_bound(weights,encoder,decoder_log_like,N_weights_enc,cur_data,samples_per_image,latent_dimensions,rs)
    lb_grad = grad(batch_value_and_grad)


    def callback(params, i, grad):
        n_iter= 0.0
        sum_ml = 0
        for j in xrange(0,10):
            ml = batch_value_and_grad(params,j)
            print "---- log marginal likelihood:", ml
            n_iter += 1
            sum_ml += ml
            print '-------- avg_ml', sum_ml/n_iter

        #Print params
        print 'norm of stdev', np.linalg.norm(np.exp(parser.get(params, 'mean')))
        print 'stepsize' , np.exp(parser.get(params,'log_stepsize'))


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

