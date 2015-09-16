# Main demo script
import pickle
import autograd.numpy as np
from autograd import grad
import time
import matplotlib.pyplot as plt

from autopaint.optimizers import adam
from autopaint.hmc import build_hmc_sampler
from autopaint.util import build_logprob_mvn,create_banded_cov
from autopaint.neuralnet import make_classification_nn
from autopaint.plotting import plot_images

param_scale = 0.1
samples_per_image = 1
latent_dimensions = 10
hidden_units = 500

if __name__ == '__main__':

    t0 = time.time()
    rs = np.random.npr.RandomState(0)

    # load_and_pickle_binary_mnist()
    with open('../../../autopaint/mnist_binary_data.pkl') as f:
        N_data, train_images, train_labels, test_images, test_labels = pickle.load(f)
    D = train_images.shape[1]

    with open('mnist_models.pkl') as f:
        trained_weights, all_mean, all_cov = pickle.load(f)

    banded_cov = create_banded_cov(all_cov.shape[0], 100)

    # Build likelihood model.
    L2_reg = 1
    layer_sizes = [784, 200, 100, 10]
    num_weights, make_predictions, likelihood = make_classification_nn(layer_sizes)
    classifier_loglik = lambda image, c: make_predictions(trained_weights, np.atleast_2d(image))[:, c]

    image_prior = build_logprob_mvn(all_mean, banded_cov)

    # Combine prior and likelihood.
    model_ll = lambda image, c: image_prior(image) + classifier_loglik(image, c)

    num_samples = 100
    num_steps = 32
    num_sampler_optimization_steps = 400
    sampler_learn_rate = 0.01

    init_mean = np.zeros(D)
    init_log_stddevs = np.log(0.1*np.ones(D))
    init_output_weights = 0.1*rs.randn(num_steps, D)
    init_transform_weights = 0.1*rs.randn(num_steps, D)
    init_biases = 0.1*rs.randn(num_steps)

    cur_class = 9
    def class_ll(image):
        return model_ll(image,cur_class)

    #flow_sample, parser = build_flow_sampler(D, num_steps)
    sampler, parser = build_hmc_sampler(class_ll, D, num_steps, leap_steps=1)

    parser.add_shape('mean',  D)
    parser.add_shape('log_stddevs', D)

    init_mean = np.zeros(D)
    init_log_stddevs = np.log(.1*np.ones(D))
    hmc_log_stepsize = np.log(.1)
    mass_mat = np.eye(D)
    v_A = np.zeros(D)
    v_B = np.zeros(D)
    v_log_cov = np.log(.01*np.ones(D))
    rev_A = np.zeros(D)
    rev_B = np.zeros(D)
    rev_log_cov = np.log(.01*np.ones(D))

    sampler_params = np.zeros(len(parser))
    parser.put(sampler_params, 'mean', init_mean)
    parser.put(sampler_params, 'log_stddev', init_log_stddevs)
    parser.put(sampler_params, 'hmc_log_stepsize', hmc_log_stepsize)
    parser.put(sampler_params, 'mass_mat', mass_mat)
    parser.put(sampler_params, 'v_A', v_A)
    parser.put(sampler_params, 'v_B', v_B)
    parser.put(sampler_params, 'v_log_cov', v_log_cov)
    parser.put(sampler_params, 'rev_A', rev_A)
    parser.put(sampler_params, 'rev_B', rev_B)
    parser.put(sampler_params, 'rev_log_cov', rev_log_cov)

    #sampler_params = np.zeros(len(parser))
    #parser.put(sampler_params, 'mean', init_mean)
    #parser.put(sampler_params, 'log_stddev', init_log_stddevs)
    #parser.put(sampler_params, 'output weights', init_output_weights)
    #parser.put(sampler_params, 'transform weights', init_transform_weights)
    #parser.put(sampler_params, 'biases', init_biases)

    def get_batch_marginal_likelihood_estimate(sampler_params, i):
        samples, entropy_estimates = sampler(sampler_params, rs, num_samples)
        likelihood_estimates = class_ll(samples)
        print "Mean loglik:", np.mean(likelihood_estimates.value),\
              "Mean entropy:", np.mean(entropy_estimates.value)

        images_per_row = 10
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
        plot_images(samples.value, ax, ims_per_row=images_per_row)
        plt.savefig('samples.png')
        return np.mean(likelihood_estimates + entropy_estimates)

    grad_func = grad(get_batch_marginal_likelihood_estimate)

    adam(grad_func, sampler_params, num_sampler_optimization_steps)
    # for i in xrange(num_sampler_optimization_steps):
    #     ml, dml = ml_and_grad(sampler_params)
    #     print "log marginal likelihood:", ml
    #     sampler_params = sampler_params + sampler_learn_rate * dml

    t1 = time.time()
    print "total runtime", t1-t0


