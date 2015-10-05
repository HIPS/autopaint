# Main demo script
import autograd.numpy as np
from autograd import value_and_grad
import numpy.linalg
import time

from autopaint.flows import build_flow_sampler_with_inputs
from autopaint.plotting import plot_density
from autopaint.util import build_logprob_mvn, WeightsParser
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

    num_samples = 500
    num_steps = 32
    num_sampler_optimization_steps = 400
    sampler_learn_rate = 0.01

    D = 2
    init_mean = np.zeros(D)
    init_log_stddevs = np.log(0.1*np.ones(D))
    init_output_weights = 0.1*rs.randn(num_steps, D)
    init_transform_weights = 0.1*rs.randn(num_steps, D)
    init_biases = 0.1*rs.randn(num_steps)

    logprob_mvn = build_logprob_mvn(mean=np.array([0.2,0.4]), cov=np.array([[1.0,0.9], [0.9,1.0]]))
    flow_sample, parser = build_flow_sampler_with_inputs(D, num_steps)
    parser.add_shape('mean',  D)
    parser.add_shape('log_stddev', D)
    sampler_params = np.zeros(len(parser))
    parser.put(sampler_params, 'mean', init_mean)
    parser.put(sampler_params, 'log_stddev', init_log_stddevs)
    parser.put(sampler_params, 'output weights', init_output_weights)
    parser.put(sampler_params, 'transform weights', init_transform_weights)
    parser.put(sampler_params, 'biases', init_biases)

    def get_batch_marginal_likelihood_estimate(sampler_params):
        mean = parser.get(sampler_params,'mean')
        log_std = parser.get(sampler_params,'log_stddev')
        samples, entropy_estimates = flow_sample(sampler_params,mean,log_std, num_samples,rs)
        likelihood_estimates = logprob_mvn(samples)
        print "Mean loglik:", np.mean(likelihood_estimates.value),\
              "Mean entropy:", np.mean(entropy_estimates.value)
        plot_density(samples.value, "approximating_dist.png")
        return np.mean(likelihood_estimates + entropy_estimates)

    ml_and_grad = value_and_grad(get_batch_marginal_likelihood_estimate)

    # Optimize Langevin parameters.
    for i in xrange(num_sampler_optimization_steps):
        ml, dml = ml_and_grad(sampler_params)
        print "log marginal likelihood:", ml
        sampler_params = sampler_params + sampler_learn_rate * dml

    t1 = time.time()
    print "total runtime", t1-t0


