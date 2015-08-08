# Script to check whether we can achieve a good lower bound on a high-dimensional
# non-spherical Gaussian prior.

import pickle
import time

import autograd.numpy as np
from autograd import value_and_grad
import matplotlib.pyplot as plt
import matplotlib.image

from autopaint.neuralnet import one_hot, train_nn, make_nn_funs, load_mnist
from autopaint.util import build_logprob_mvt, build_logprob_mvn, entropy_of_a_gaussian,\
    sigmoid, inv_sigmoid, mean_and_cov
from autopaint.plotting import plot_images
from autopaint.inference import build_langevin_sampler


def train_mnist_model():
    # Load and process MNIST data
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    trained_weights, predict_fun,likeFunc = train_nn(train_images, train_labels, test_images, test_labels)
    all_mean, all_cov = mean_and_cov(train_images)
    mnist_models = trained_weights, all_mean, all_cov
    with open('mnist_models.pkl', 'w') as f:
        pickle.dump(mnist_models, f, 1)


def plot_sampler_params(params):

    mean = parser.get(params, 'mean')
    stddev = np.exp(parser.get(params, 'log_stddev'))
    stepsizes = np.exp(parser.get(params, 'log_stepsizes'))
    noise_sizes = np.exp(parser.get(params, 'log_noise_sizes'))
    gradscales = np.exp(parser.get(params, 'log_gradient_scales'))

    fig = plt.figure(0)
    fig.clf()

    ax = fig.add_subplot(411)
    ax.plot(stepsizes, 'o-')
    ax.set_ylabel('stepsizes', fontproperties='serif')

    ax = fig.add_subplot(412)
    ax.plot(noise_sizes, 'o-')
    ax.set_ylabel('noise sizes', fontproperties='serif')

    fig.subplots_adjust(hspace=.5)
    plt.savefig('params.png')

    matplotlib.image.imsave("mean.png", mean.reshape((28,28)))
    matplotlib.image.imsave("stddev.png", stddev.reshape((28,28)))
    matplotlib.image.imsave("gradscale.png", gradscales.reshape((28,28)))


def plot_sampler_param_grads(params):

    mean = parser.get(params, 'mean')
    stddev = parser.get(params, 'log_stddev')
    stepsizes = parser.get(params, 'log_stepsizes')
    noise_sizes = parser.get(params, 'log_noise_sizes')
    gradscales = parser.get(params, 'log_gradient_scales')

    fig = plt.figure(0)
    fig.clf()

    ax = fig.add_subplot(411)
    ax.plot(stepsizes, 'o-')
    ax.set_ylabel('grad of log stepsizes', fontproperties='serif')

    ax = fig.add_subplot(412)
    ax.plot(noise_sizes, 'o-')
    ax.set_ylabel('grad of log noise sizes', fontproperties='serif')

    fig.subplots_adjust(hspace=.5)
    plt.savefig('grad_params.png')

    matplotlib.image.imsave("grad_init_mean.png", mean.reshape((28,28)))
    matplotlib.image.imsave("grad_stddev.png", stddev.reshape((28,28)))
    matplotlib.image.imsave("grad_gradscales.png", gradscales.reshape((28,28)))


if __name__ == '__main__':

    start_time = time.time()

    num_samples = 20
    num_langevin_steps = 5
    num_sampler_optimization_steps = 300
    sampler_learn_rate = 0.01
    images_per_row = 10

    init_init_stddev_scale = 0.1
    init_langevin_stepsize = 0.00001
    init_langevin_noise_size = 0.00001
    init_gradient_power = 0.95
    t_degrees_of_freedom = 20.0

    layer_sizes = [784, 200, 100, 10]
    L2_reg = 1.0
    D = 784
    N_weights, predict_fun, loss_fun, frac_err, nn_loglik = make_nn_funs(layer_sizes, L2_reg)

    with open('mnist_models.pkl') as f:
        trained_weights, all_mean, all_cov = pickle.load(f)

    # Regularize all_cov
    prior_relax = 0.05
    all_cov = all_cov + prior_relax * np.eye(D)

    prior_func = build_logprob_mvt(all_mean, all_cov, t_degrees_of_freedom)
    print "Prior entropy:", entropy_of_a_gaussian(all_cov)

    def nn_likelihood(images, labels):
        prior = prior_func(images)
        squashed_images = sigmoid(images)
        likelihood = nn_loglik(trained_weights, squashed_images, labels)
        return prior + likelihood

    gen_labels = one_hot(np.array([i % 10 for i in range(num_samples)]), 10)
    labeled_likelihood = lambda images: nn_likelihood(images, gen_labels)

    init_mean = all_mean
    init_stddevs = np.log(init_init_stddev_scale * np.ones((1,D)))
    init_log_stepsizes = np.log(init_langevin_stepsize * np.ones(num_langevin_steps))
    init_log_noise_sizes = np.log(init_langevin_noise_size * np.ones(num_langevin_steps))
    init_log_gradient_scales = np.log(np.ones((1,D)))

    sample_and_run_langevin, parser = build_langevin_sampler(labeled_likelihood, D,  num_langevin_steps, approx=True)

    sampler_params = np.zeros(len(parser))
    parser.put(sampler_params, 'mean', init_mean)
    parser.put(sampler_params, 'log_stddev', init_stddevs)
    parser.put(sampler_params, 'log_stepsizes', init_log_stepsizes)
    parser.put(sampler_params, 'log_noise_sizes', init_log_noise_sizes)
    parser.put(sampler_params, 'log_gradient_scales', init_log_gradient_scales)
    parser.put(sampler_params, 'invsig_gradient_power', inv_sigmoid(init_gradient_power))

    rs = np.random.npr.RandomState(0)
    def batch_marginal_likelihood_estimate(sampler_params):
        samples, likelihood_estimates, entropy_estimates = sample_and_run_langevin(sampler_params, rs, num_samples)
        print "Mean loglik:", np.mean(likelihood_estimates.value), "Mean entropy:", np.mean(entropy_estimates.value)
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
        plot_images(samples.value, ax, ims_per_row=images_per_row)
        plt.savefig('samples.png')

        return np.mean(likelihood_estimates + entropy_estimates)

    ml_and_grad = value_and_grad(batch_marginal_likelihood_estimate)

    # Optimize Langevin parameters.
    for i in xrange(num_sampler_optimization_steps):
        ml, dml = ml_and_grad(sampler_params)
        print "Iter:", i, "log marginal likelihood:", ml, "avg gradient size: ", np.mean(np.abs(dml))
        print "Gradient power:", sigmoid(parser.get(sampler_params, 'invsig_gradient_power'))
        plot_sampler_params(sampler_params)
        plot_sampler_param_grads(dml)

        sampler_params = sampler_params + sampler_learn_rate * dml

    finished_time = time.time()
    print "Total runtime:", finished_time - start_time


