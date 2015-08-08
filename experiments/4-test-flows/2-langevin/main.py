# Main demo script
import autograd.numpy as np
from autograd import value_and_grad
import numpy.linalg
import time

from autopaint.inference import build_langevin_sampler
from autopaint.plotting import plot_density
from autopaint.util import build_logprob_mvn, sigmoid, inv_sigmoid, logprob_two_moons

def logprob_two_moons(z):
    z1 = z[:, 0]
    z2 = z[:, 1]
    return (- 0.5 * ((np.sqrt(z1**2 + z2**2) - 2 ) / 0.4)**2\
            + np.logaddexp(-0.5 * ((z1 - 2) / 0.6)**2, -0.5 * ((z1 + 2) / 0.6)**2))

def logprob_wiggle(z):
    z1 = z[:, 0]
    z2 = z[:, 1]
    return -0.5 * (z2 - np.sin(2.0 * np.pi * z1 / 4.0) / 0.4 )**2 - 0.2 * (z1**2 + z2**2)

cov = np.array([[1.0, 0.9], [0.9, 1.0]])
pinv = np.linalg.pinv(cov)
(sign, logdet) = numpy.linalg.slogdet(cov)
const =  -0.5 * 2 * np.log(2*np.pi) - 0.5 * logdet
def logprob_mvn(z):
    return const - 0.5 * np.dot(np.dot(z.T, pinv), z)


if __name__ == '__main__':

    t0 = time.time()
    rs = np.random.npr.RandomState(0)

    num_samples = 200
    num_steps = 5
    num_sampler_optimization_steps = 400
    sampler_learn_rate = 0.01

    init_init_stddev_scale = 1.0
    init_langevin_stepsize = 0.1
    init_langevin_noise_size = 0.001
    init_gradient_power = 0.95

    D = 2
    init_mean = np.zeros(D)
    init_stddevs = np.log(init_init_stddev_scale * np.ones((1,D)))
    init_log_stepsizes = np.log(init_langevin_stepsize * np.ones(num_steps))
    init_log_noise_sizes = np.log(init_langevin_noise_size * np.ones(num_steps))
    init_log_gradient_scales = np.log(np.ones((1,D)))

    logprob_mvn = build_logprob_mvn(mean=np.array([0.2,0.4]), cov=np.array([[1.0,0.9], [0.9,1.0]]))
    sample, parser = build_langevin_sampler(logprob_two_moons, D, num_steps, approx=False)

    sampler_params = np.zeros(len(parser))
    parser.put(sampler_params, 'mean', init_mean)
    parser.put(sampler_params, 'log_stddev', init_stddevs)
    parser.put(sampler_params, 'log_stepsizes', init_log_stepsizes)
    parser.put(sampler_params, 'log_noise_sizes', init_log_noise_sizes)
    parser.put(sampler_params, 'log_gradient_scales', init_log_gradient_scales)
    parser.put(sampler_params, 'invsig_gradient_power', inv_sigmoid(init_gradient_power))

    def get_batch_marginal_likelihood_estimate(sampler_params):
        samples, likelihood_estimates, entropy_estimates = sample(sampler_params, rs, num_samples)
        print "Mean loglik:", np.mean(likelihood_estimates.value),\
              "Mean entropy:", np.mean(entropy_estimates.value)
        plot_density(samples.value, "approximating_dist.png")
        return np.mean(likelihood_estimates + entropy_estimates)

    ml_and_grad = value_and_grad(get_batch_marginal_likelihood_estimate)

    # Optimize Langevin parameters.
    for i in xrange(num_sampler_optimization_steps):
        ml, dml = ml_and_grad(sampler_params)
        print "Iter:", i, "log marginal likelihood:", ml, "avg gradient size: ", np.mean(np.abs(dml))
        print "Gradient power:", sigmoid(parser.get(sampler_params, 'invsig_gradient_power'))
        sampler_params = sampler_params + sampler_learn_rate * dml

    t1 = time.time()
    print "total runtime", t1-t0


