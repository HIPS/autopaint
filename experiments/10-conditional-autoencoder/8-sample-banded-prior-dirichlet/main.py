import pickle
import time
import matplotlib.pyplot as plt

import autograd.numpy.random as npr
import autograd.numpy as np

from autograd import grad

from autopaint.plotting import plot_images
from autopaint.optimizers import adam
from autopaint.neuralnet import make_batches
from autopaint.util import load_mnist,create_banded_cov,build_logprob_mvn
from autopaint.aevb_ent import enc_lower_bound
from autopaint.util import WeightsParser, load_and_pickle_binary_mnist,build_logprob_standard_normal
from autopaint.neuralnet import make_gaussian_nn, make_classification_nn
param_scale = 0.1
samples_per_image = 10
latent_dimensions = 784
hidden_units = 500

def run_variational_network(train_images,N_weights_dec,decoder,decoder_log_like,trained_weights,all_mean,all_cov):
    start_time = time.time()

    # Create aevb function
    # Training parameters

    D = train_images.shape[1]

    enc_layers = [D, hidden_units,hidden_units, 2*latent_dimensions]

    N_weights_enc, encoder, encoder_log_like = make_gaussian_nn(enc_layers)

    # Optimize aevb
    batch_size = 10
    num_training_iters = 1600
    rs = npr.RandomState(0)

    parser = WeightsParser()
    parser.add_shape('encoding weights', (N_weights_enc,))
    initial_enc_w = rs.randn(len(parser)) * param_scale

    batch_idxs = make_batches(train_images.shape[0], batch_size)
    banded_cov = create_banded_cov(all_cov.shape[0],10)
    log_prior = build_logprob_mvn(all_mean, banded_cov)
    def batch_value_and_grad(enc_w, iter):
        iter = iter % len(batch_idxs)
        cur_data = train_images[batch_idxs[iter]]
        return enc_lower_bound(enc_w,trained_weights,encoder,decoder_log_like,log_prior,N_weights_enc,cur_data,samples_per_image,latent_dimensions,rs)
    lb_grad = grad(batch_value_and_grad)

    def callback(params, i, grad):
        ml = batch_value_and_grad(params,i)
        print "log marginal likelihood:", ml
        #Generate samples
        num_samples = 100
        images_per_row = 10
        # zs = train_images[0:100,:]
        zs = np.random.dirichlet(.1*np.ones(10),100)
        (mus,log_sigs) = encoder(params, zs)
        # sigs = np.exp(log_sigs)
        # noise = rs.randn(1,100,784)
        # samples = mus + sigs*noise
        # samples = np.reshape(samples,(100*1,784),order = 'F')
        samples = mus

        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
        plot_images(samples, ax, ims_per_row=images_per_row)
        plt.savefig('samples.png')

    final_params = adam(lb_grad, initial_enc_w, num_training_iters, callback=callback)

    def decoder_with_weights(zs):
        return decoder(parser.get(final_params, 'decoding weights'), zs)
    return decoder_with_weights

    finish_time = time.time()
    print "total runtime", finish_time - start_time




if __name__ == '__main__':
    # load_and_pickle_binary_mnist()
    with open('../../../autopaint/mnist_binary_data.pkl') as f:
        N_data, train_images, train_labels, test_images, test_labels = pickle.load(f)

    # Create aevb function
    # Training parameters

    D = train_images.shape[1]

    enc_layers = [D, hidden_units, 2*latent_dimensions]
    dec_layers = [latent_dimensions, hidden_units, D]

    N_weights_enc, encoder, encoder_log_like = make_gaussian_nn(enc_layers)


    with open('mnist_models.pkl') as f:
        trained_weights, all_mean, all_cov = pickle.load(f)


   # Build likelihood model.
    L2_reg = 1
    layer_sizes = [784, 200, 100, 10]
    N_weights_dec, decoder, decoder_log_like = make_classification_nn(layer_sizes)


    sampled_labels = np.random.dirichlet(.1*np.ones(10),train_images.shape[0])

    decoder = run_variational_network(sampled_labels,N_weights_dec,decoder,decoder_log_like,trained_weights,all_mean,all_cov)

