# Main demo script
import autograd.numpy as np
from autograd import value_and_grad
import numpy.linalg
import matplotlib.image
import pickle
import time

from autograd import grad

import matplotlib.pyplot as plt

from autopaint.plotting import *
from autopaint.early_stop import build_early_stop
from autopaint.util import build_logprob_mvn

def plot_sampler_params(params, filename):

    mean = parser.get(params, 'mean')
    stddev = np.exp(parser.get(params, 'log_stddev'))
    stepsizes = np.exp(parser.get(params, 'log_stepsize'))


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
    ax.set_xlabel('Langevin iterations', fontproperties='serif')


    fig.subplots_adjust(hspace=.5)
    plt.savefig(filename)


if __name__ == '__main__':

    t0 = time.time()

    num_samples = 100
    num_sampler_optimization_steps = 200
    sampler_learn_rate = .2

    D = 2
    init_mean = np.zeros(D)
    init_log_stddevs = np.log(1.0*np.ones(D))
    init_log_stepsize = np.log(0.001)

    rs = np.random.npr.RandomState(0)
    logprob_mvn = build_logprob_mvn(np.zeros(2),np.array([[1,.9],[.9,1]]))
    sample_and_run_early_stop, parser = build_early_stop(logprob_mvn, D, approx=False)

    sampler_params = np.zeros(len(parser))
    parser.put(sampler_params, 'mean', init_mean)
    parser.put(sampler_params, 'log_stddev', init_log_stddevs)
    parser.put(sampler_params, 'log_stepsize', init_log_stepsize)



    def get_batch_marginal_likelihood_estimate(sampler_params):
        # loglik_estimates = np.array(num_samples)
        # entropy_estimates = np.array(num_samples)
        # samples = []
        # loglik_estimates = []
        # entropy_estimates = []
        for i in xrange(num_samples):
            sample, loglik_estimate, entropy_estimate =   sample_and_run_early_stop(sampler_params,rs,1)
            if i == 0:
                samples = sample
                loglik_estimates = loglik_estimate
                entropy_estimates = entropy_estimate
            else:
                samples = np.concatenate((samples,sample),axis = 0)
                loglik_estimates = np.concatenate((loglik_estimates, loglik_estimate),axis = 0)
                entropy_estimates = np.concatenate((entropy_estimates, entropy_estimate),axis = 0)
        marginal_likelihood_estimates = loglik_estimates + entropy_estimates
        print "mean loglik:", np.mean(loglik_estimates), " mean entropy:", np.mean(entropy_estimates)
        plot_density(samples.value, "approximating_dist.png")
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


