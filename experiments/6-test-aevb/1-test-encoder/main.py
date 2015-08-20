import pickle
import time
import matplotlib.pyplot as plt

import autograd.numpy.random as npr
import autograd.numpy as np

from autograd import value_and_grad

from autopaint.plotting import plot_images
from autopaint.optimizers import adam_mini_batch
from autopaint.neuralnet import make_batches
from autopaint.util import load_mnist
from autopaint.aevb import build_encoder, build_binarized_decoder, lower_bound

def run_aevb(train_images):
    start_time = time.time()

    #Create aevb function
    # Training parameters
    param_scale = 0.1
    samples_per_image = 5
    latent_dimensions = 20
    hidden_units = 500
    D = train_images.shape[1]

    enc_layers = [D, hidden_units, 2*latent_dimensions]
    dec_layers = [latent_dimensions, hidden_units, D]

    N_weights_enc,encoder = build_encoder(enc_layers)
    N_weights_dec,decoder, decoder_log_like = build_binarized_decoder(dec_layers)

    #Optimize aevb
    batch_size = 256
    num_epochs = 160
    rs = npr.RandomState([0])
    enc_w = rs.randn(N_weights_enc) * param_scale
    dec_w = rs.randn(N_weights_dec) * param_scale

    weights = np.concatenate((enc_w,dec_w))

    batch_idxs = make_batches(train_images.shape[0], batch_size)

    def batch_value_and_grad(weights,idxs):
        def batch_lower_bound(weights):
            return lower_bound(weights,encoder,decoder_log_like,N_weights_enc,train_images[idxs],samples_per_image,latent_dimensions,rs)
        #TODO:Make it so we don't have to recompute gradient at each iter? (Currently not necessary since it is very fast)
        val_and_grad_func = value_and_grad(batch_lower_bound)
        #Generate samples
        num_samples = 100
        images_per_row = 10
        z = rs.randn(num_samples,latent_dimensions)
        samples = decoder(dec_w,z)
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
        plot_images(samples, ax, ims_per_row=images_per_row)
        plt.savefig('samples.png')

        return val_and_grad_func(weights)

    def print_ml(ml,weights,grad):
        print "log marginal likelihood:", ml

    final_params, final_value = adam_mini_batch(batch_value_and_grad,weights,batch_idxs,num_epochs,callback=print_ml)

    final_dec_w = final_params[N_weights_enc:len(weights)]

    # #Generate samples
    # num_samples = 100
    # z = rs.randn(num_samples,latent_dimensions)
    # muSamples,log_sig_samples = decoder(final_dec_w,z)
    # sig_samples = np.exp(log_sig_samples)
    # noise = rs.randn(num_samples,D)
    # samples = muSamples+sig_samples*noise
    # print 'empirical mean', np.mean(samples,axis=0)
    # print 'empirical cov', np.cov((samples.T))
    # plot_density(samples, "aevb_approximating_dist.png")


    finish_time = time.time()
    print "total runtime", finish_time - start_time


def load_and_pickle_binary_mnist():
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    train_images = np.round(train_images)
    test_images = np.round(test_images)
    mnist_data = N_data, train_images, train_labels, test_images, test_labels
    with open('mnist_binary_data.pkl', 'w') as f:
        pickle.dump(mnist_data, f, 1)

if __name__ == '__main__':
    #load_and_pickle_binary_mnist()
    with open('mnist_binary_data.pkl') as f:
        N_data, train_images, train_labels, test_images, test_labels = pickle.load(f)


    # train_images = np.random.multivariate_normal(np.zeros(2),np.array([[1,0],[0,1]]),1000)
    run_aevb(train_images)