# Main demo script
import sys
sys.path.append('../../autopaint/')
import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
import numpy.linalg
import matplotlib.image
import pickle
import time
from autopaint.neuralnet import *

from autograd import grad

import matplotlib.pyplot as plt

from autopaint.util import fast_array_from_list, load_mnist, mean_and_cov
from autopaint.plotting import *
from autopaint.inference import build_langevin_sampler


# from util import fast_array_from_list
# from plotting import *
# from inference import build_langevin_sampler


def model_mnist():
    # Load and process MNIST data
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    trained_weights, predict_fun,likeFunc = train_nn(train_images, train_labels, test_images, test_labels)
    all_mean, all_cov = mean_and_cov(train_images)
    mnist_models = trained_weights, all_mean, all_cov
    with open('mnist_models.pkl', 'w') as f:
        pickle.dump(mnist_models, f, 1)

def logprob_two_moons(z):
    z1 = z[0]
    z2 = z[1]
    return (- 0.5 * ((np.sqrt(z1**2 + z2**2) - 2 ) / 0.4)**2\
            + np.logaddexp(-0.5 * ((z1 - 2) / 0.6)**2, -0.5 * ((z1 + 2) / 0.6)**2))

def logprob_wiggle(z):
    z1 = z[0]
    z2 = z[1]
    return -0.5 * (z2 - np.sin(2.0 * np.pi * z1 / 4.0) / 0.4 )**2 + 0.2 * (z1**2 + z2**2)

# cov = np.array([[1.0, 0.9], [0.9, 1.0]])

def logprob_mvn(mean,cov,z):
    #TODO: Precompute pinv,logdet
    if z.ndim == 1:
        z = np.reshape(z,(1,len(z)))
    num_points = z.shape[0]
    mean_mat = np.tile(mean,(num_points,1))
    pinv = np.linalg.pinv(cov)
    (sign, logdet) = numpy.linalg.slogdet(cov)
    const =  -0.5 * 2 * np.log(2*np.pi) - 0.5 * logdet
    return num_points*const - 0.5 * np.dot(np.dot((z-mean_mat), pinv), (z-mean_mat).T)

def plot_sampler_params(params, filename):

    mean = parser.get(params, 'mean')
    stddev = parser.get(params, 'log_stddev')
    stepsizes = np.exp(parser.get(params, 'log_stepsizes'))
    noise_sizes = np.exp(parser.get(params, 'log_noise_sizes'))

    # ----- Nice versions of Alpha and beta schedules for paper -----
    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(411)
    ax.plot(mean, 'o-')
    ax.set_ylabel('Mean', fontproperties='serif')

    ax = fig.add_subplot(412)
    ax.plot(stddev, 'o-')
    ax.set_ylabel('stddev', fontproperties='serif')

    ax = fig.add_subplot(413)
    ax.plot(stepsizes, 'o-')
    ax.set_ylabel('stepsizes', fontproperties='serif')

    ax = fig.add_subplot(414)
    ax.plot(noise_sizes, 'o-')
    ax.set_ylabel('noise_sizes', fontproperties='serif')

    plt.savefig(filename)


if __name__ == '__main__':

    t0 = time.time()
    num_samples = 40
    num_langevin_steps = 10
    num_sampler_optimization_steps = 20
    sampler_learn_rate = 0.001

    # model_mnist()

    with open('mnist_models.pkl') as f:
        trained_weights, all_mean, all_cov = pickle.load(f)

    layer_sizes = [784, 200, 100, 10]
    L2_reg = 1.0

    N_weights, predict_fun, loss_fun, frac_err, nn_like = make_nn_funs(layer_sizes, L2_reg)


    def generative_conditional(labels):
        def cond_like(images):
            return nn_like(trained_weights,images,labels)+logprob_mvn(all_mean,all_cov,images)
        return cond_like

    labels = np.zeros((num_samples,1))

    cond_like = generative_conditional(labels)

    D = 784
    init_mean = all_mean
    # init_stddevs = np.diag(all_cov)+.1
    # init_mean = np.zeros((1,D))
    init_stddevs = np.log(.00001*np.ones((1,D)))
    init_log_stepsizes = np.log(0.001*np.ones(num_langevin_steps))
    init_log_noise_sizes = np.log(.0000001*np.ones(num_langevin_steps))

    rs = np.random.npr.RandomState(0)

    sample_and_run_langevin, parser = build_langevin_sampler(cond_like, D, num_langevin_steps,approx = True)

    sampler_params = np.zeros(len(parser))
    parser.put(sampler_params, 'mean', init_mean)
    parser.put(sampler_params, 'log_stddev', init_stddevs)
    parser.put(sampler_params, 'log_stepsizes', init_log_stepsizes)
    parser.put(sampler_params, 'log_noise_sizes', init_log_noise_sizes)
    #
    # def get_batch_marginal_likelihood_estimate(sampler_params):
    #     samples = [] # np.zeros((num_samples, D))
    #     mls = []
    #     for s in range(num_samples):
    #         z, marginal_likelihood_estimate = sample_and_run_langevin(sampler_params, rs)
    #         samples.append(z.value)
    #         mls.append(marginal_likelihood_estimate)
    #
    #     samples = fast_array_from_list(samples)
    #     # plot_density(samples, "approximating_dist.png")
    #     matplotlib.image.imsave("optimizing", samples[0,:].reshape((28,28)))
    #
    #     return np.mean(mls)

    def get_batch_marginal_likelihood_estimate(sampler_params):
        samples, marginal_likelihood_estimates = sample_and_run_langevin(sampler_params, rs, num_samples)
        matplotlib.image.imsave("optimizing", (samples[0,:].reshape((28,28))).value)

        return np.mean(marginal_likelihood_estimates)

    ml_and_grad = value_and_grad(get_batch_marginal_likelihood_estimate)

    # Optimize Langevin parameters.
    for i in xrange(num_sampler_optimization_steps):
        ml, dml = ml_and_grad(sampler_params)
        print "log marginal likelihood:", ml
        plot_sampler_params(sampler_params, 'sampler_params.png')
        sampler_params = sampler_params + sampler_learn_rate * dml



    t1 = time.time()
    print "total runtime", t1-t0


