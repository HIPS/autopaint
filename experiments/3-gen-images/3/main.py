# Main demo script
import pickle
import time

import autograd.numpy as np
from autograd import value_and_grad
import matplotlib.pyplot as plt
import matplotlib.image

from autopaint.neuralnet import one_hot, train_nn, make_nn_funs
from autopaint.util import load_mnist, mean_and_cov, build_logprob_mvn
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

def plot_sampler_params(params, filename_prefix):

    mean = parser.get(params, 'mean')
    stddev = np.exp(parser.get(params, 'log_stddev'))
    stepsizes = np.exp(parser.get(params, 'log_stepsizes'))
    noise_sizes = np.exp(parser.get(params, 'log_noise_sizes'))

    # ----- Nice versions of Alpha and beta schedules for paper -----
    fig = plt.figure(0)
    fig.clf()

    ax = fig.add_subplot(411)
    ax.plot(stepsizes, 'o-')
    ax.set_ylabel('stepsizes', fontproperties='serif')

    ax = fig.add_subplot(412)
    ax.plot(noise_sizes, 'o-')
    ax.set_ylabel('noise_sizes', fontproperties='serif')

    fig.subplots_adjust(hspace=.5)
    plt.savefig(filename_prefix + '.png')

    matplotlib.image.imsave(filename_prefix + "_mean.png", mean.reshape((28,28)))
    matplotlib.image.imsave(filename_prefix + "_stddev.png", np.exp(stddev.reshape((28,28))))


if __name__ == '__main__':

    t0 = time.time()

    num_samples = 15
    num_langevin_steps = 1
    num_sampler_optimization_steps = 20
    sampler_learn_rate = 0.001

    layer_sizes = [784, 200, 100, 10]
    L2_reg = 1.0
    D = 784

    init_init_stddev_scale = 0.0001
    init_langevin_stepsize = 0.0001
    init_langevin_noise_size = 0.0001

    # train_mnist_model()   # Comment after running once.

    with open('mnist_models.pkl') as f:
        trained_weights, all_mean, all_cov = pickle.load(f)

    N_weights, predict_fun, loss_fun, frac_err, nn_like = make_nn_funs(layer_sizes, L2_reg)

    prior_func = build_logprob_mvn(all_mean, all_cov)
    def nn_likelihood(images, labels):
        prior = prior_func(images)
        likelihood = nn_like(trained_weights,images,labels)
        return prior + likelihood

    gen_labels = one_hot(np.array([i % 10 for i in range(num_samples)]), 10)
    labeled_likelihood = lambda images: nn_likelihood(images, gen_labels)

    init_mean = all_mean
    # init_stddevs = np.diag(all_cov)+.1
    # init_mean = np.zeros((1,D))
    init_stddevs = np.log(init_init_stddev_scale * np.ones((1,D)))
    init_log_stepsizes = np.log(init_langevin_stepsize * np.ones(num_langevin_steps))
    init_log_noise_sizes = np.log(init_langevin_noise_size * np.ones(num_langevin_steps))

    sample_and_run_langevin, parser = build_langevin_sampler(labeled_likelihood, D,
                                                             num_langevin_steps, approx=True)

    sampler_params = np.zeros(len(parser))
    parser.put(sampler_params, 'mean', init_mean)
    parser.put(sampler_params, 'log_stddev', init_stddevs)
    parser.put(sampler_params, 'log_stepsizes', init_log_stepsizes)
    parser.put(sampler_params, 'log_noise_sizes', init_log_noise_sizes)

    rs = np.random.npr.RandomState(0)
    def batch_marginal_likelihood_estimate(sampler_params):
        samples, likelihood_estimates, entropy_estimates = sample_and_run_langevin(sampler_params, rs, num_samples)
        #matplotlib.image.imsave("optimizing", (samples[0,:].reshape((28,28))).value)
        #marginal_likelihood_estimates
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
        plot_images(samples.value, ax)
        plt.savefig('samples.png')

        return np.mean(likelihood_estimates + entropy_estimates)

    ml_and_grad = value_and_grad(batch_marginal_likelihood_estimate)

    # Optimize Langevin parameters.
    for i in xrange(num_sampler_optimization_steps):
        ml, dml = ml_and_grad(sampler_params)
        print "iter:", i, "log marginal likelihood:", ml
        plot_sampler_params(sampler_params, 'params')
        plot_sampler_params(dml, 'param_grads')

        sampler_params = sampler_params + sampler_learn_rate * dml


    t1 = time.time()
    print "total runtime", t1-t0


