import pickle
import time
import matplotlib.pyplot as plt

import autograd.numpy.random as npr
import autograd.numpy as np

from autograd import grad,value_and_grad
from autograd.util import quick_grad_check
import matplotlib
from scipy.optimize import minimize

from autopaint.plotting import plot_images
from autopaint.optimizers import adam
from autopaint.neuralnet import make_batches,make_nn_funs
from autopaint.util import load_mnist
from autopaint.aevb import lower_bound
from autopaint.util import WeightsParser, load_and_pickle_binary_mnist
from autopaint.neuralnet import make_binary_nn,make_gaussian_nn
param_scale = 0.1
samples_per_image = 1
latent_dimensions = 2
hidden_units = 500

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
    num_training_iters = 16
    rs = npr.RandomState(0)

    parser = WeightsParser()
    parser.add_shape('encoding weights', (N_weights_enc,))
    parser.add_shape('decoding weights', (N_weights_dec,))
    initial_combined_weights = rs.randn(len(parser)) * param_scale

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
        samples = decoder(parser.get(params, 'decoding weights'), zs)
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
        plot_images(samples, ax, ims_per_row=images_per_row)
        plt.savefig('samples.png')

    final_params = adam(lb_grad, initial_combined_weights, num_training_iters, callback=callback)

    def lb_fun(data):
        lower_bound(final_params,encoder,decoder_log_like,N_weights_enc,data,samples_per_image,latent_dimensions,rs)

    with open('lb_fun.pkl', 'w') as f:
        pickle.dump(lb_fun, f, 1)

    finish_time = time.time()
    print "total runtime", finish_time - start_time

if __name__ == '__main__':
    # load_and_pickle_binary_mnist()
    with open('../../../autopaint/mnist_binary_data.pkl') as f:
        N_data, train_images, train_labels, test_images, test_labels = pickle.load(f)

    run_aevb(train_images)
    with open('lb_fun.pkl') as f:
        lb_fun = pickle.load(f)

    with open('mnist_models.pkl') as f:
        trained_weights, all_mean, all_cov = pickle.load(f)

   # Build likelihood model.
    L2_reg = 1
    layer_sizes = [784, 200, 100, 10]
    N_weights, predict_fun, loss_fun, frac_err = make_nn_funs(layer_sizes, L2_reg)
    classifier_loglik = lambda image, c: predict_fun(trained_weights, np.atleast_2d(image))[:, c]

    # Combine prior and likelihood.
    model_ll = lambda image, c: lb_fun(image) +classifier_loglik(image, c)
    model_ll_with_grad = value_and_grad(model_ll)

   # Optimize a random image to maximize this likelihood.
    cur_class = 9
    start_image = np.zeros((28*28))
    quick_grad_check(lb_fun, start_image)

    def callback(image):
        #print "Cur loglik: ", image_prior_nll(image), "mean loglik:", image_prior_nll(all_mean)
        matplotlib.image.imsave("../figures/optimizing", image.reshape((28,28)))

    # Optimize using conjugate gradients.
    result = minimize(-model_nll_with_grad, callback=callback, x0=start_image, args=(cur_class),
                      jac=True, method='BFGS')
    final_image = result.x
    matplotlib.image.imsave("../figures/optimal", final_image.reshape((28,28)))
    print "Finished!"
