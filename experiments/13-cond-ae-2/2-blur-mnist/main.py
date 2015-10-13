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
from autopaint.cond_aevb import lower_bound
from autopaint.util import WeightsParser, load_and_pickle_binary_mnist
from autopaint.neuralnet import make_binary_nn,make_gaussian_nn
param_scale = 0.1
samples_per_image = 10
latent_dimensions = 10
hidden_units = 500

from scipy.ndimage.filters import gaussian_filter

def run_cond_aevb(base_data,cond_data):
    start_time = time.time()

    # Create aevb function
    # Training parameters


    D_c = cond_data.shape[1]
    D_b = base_data.shape[1]
    N_data = cond_data.shape[0]
    assert cond_data.shape[0] == base_data.shape[0]
    enc_layers = [D_c, hidden_units, 2*latent_dimensions]
    dec_layers = [latent_dimensions+D_b, hidden_units, D_c]

    N_weights_enc, encoder, encoder_log_like = make_gaussian_nn(enc_layers)
    N_weights_dec, decoder, decoder_log_like = make_binary_nn(dec_layers)

    # Optimize aevb
    batch_size = 100
    num_training_iters = 1600
    rs = npr.RandomState(0)

    parser = WeightsParser()
    parser.add_shape('encoding weights', (N_weights_enc,))
    parser.add_shape('decoding weights', (N_weights_dec,))
    initial_combined_weights = rs.randn(len(parser)) * param_scale

    batch_idxs = make_batches(N_data, batch_size)

    def batch_value_and_grad(weights, iter):
        iter = iter % len(batch_idxs)
        cur_cond = cond_data[batch_idxs[iter]]
        cur_base = base_data[batch_idxs[iter]]
        return lower_bound(weights,encoder,decoder_log_like,N_weights_enc,cur_base,cur_cond,samples_per_image,latent_dimensions,rs)

    lb_grad = grad(batch_value_and_grad)



    def callback(params, i, grad):
        ml = batch_value_and_grad(params,i)
        print "log marginal likelihood:", ml

        # #Generate samples
        num_samples = 100
        images_per_row = 10
        zs = rs.randn(num_samples,latent_dimensions)
        base_test = base_data[0:num_samples,:]
        dec_in = np.concatenate((zs,base_test),axis = 1)
        samples = decoder(parser.get(params, 'decoding weights'), dec_in)
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
        plot_images(samples, ax, ims_per_row=images_per_row)
        plt.savefig('samples.png')
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
        plot_images(base_test, ax, ims_per_row=images_per_row)
        plt.savefig('blurred_samples.png')

    final_params = adam(lb_grad, initial_combined_weights, num_training_iters, callback=callback)

    def decoder_with_weights(zs):
        return decoder(parser.get(final_params, 'decoding weights'), zs)
    return decoder_with_weights

    finish_time = time.time()
    print "total runtime", finish_time - start_time




if __name__ == '__main__':
    # load_and_pickle_binary_mnist()
    with open('../../../autopaint/mnist_binary_data.pkl') as f:
        N_data, train_im, train_l, test_im, test_l = pickle.load(f)
    base_data = gaussian_filter(train_im,1)
    cond_data = train_im
    decoder = run_cond_aevb(base_data,cond_data)
