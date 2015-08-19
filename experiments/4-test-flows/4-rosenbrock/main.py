# Main demo script
import autograd.numpy as np
from autograd import value_and_grad
import numpy.linalg
import time

from autopaint.flows import build_flow_sampler
from autopaint.plotting import plot_density
from autopaint.util import build_logprob_mvn, log_inv_rosenbrock, log_tapered_inv_rosenbrock
from autopaint.optimizers import adam

cov = np.array([[1.0, 0.9], [0.9, 1.0]])
pinv = np.linalg.pinv(cov)
(sign, logdet) = numpy.linalg.slogdet(cov)
const =  -0.5 * 2 * np.log(2*np.pi) - 0.5 * logdet
def logprob_mvn(z):
    return const - 0.5 * np.dot(np.dot(z.T, pinv), z)


if __name__ == '__main__':

    t0 = time.time()
    rs = np.random.npr.RandomState(0)

    num_samples = 50
    num_steps = 10
    num_sampler_optimization_steps = 400

    D = 2
    init_mean = np.zeros(D)
    init_log_stddevs = np.log(10*np.ones(D))
    init_output_weights = 0.1*rs.randn(num_steps, D)
    init_transform_weights = 0.1*rs.randn(num_steps, D)
    init_biases = 0.1*rs.randn(num_steps)

    #logprob_mvn = build_logprob_mvn(mean=np.array([0.2,0.4]), cov=np.array([[1.0,0.9], [0.9,1.0]]))
    flow_sample, parser = build_flow_sampler(log_tapered_inv_rosenbrock, D, num_steps)

    sampler_params = np.zeros(len(parser))
    parser.put(sampler_params, 'mean', init_mean)
    parser.put(sampler_params, 'log_stddev', init_log_stddevs)
    parser.put(sampler_params, 'output weights', init_output_weights)
    parser.put(sampler_params, 'transform weights', init_transform_weights)
    parser.put(sampler_params, 'biases', init_biases)

    def get_batch_marginal_likelihood_estimate(sampler_params):
        samples, likelihood_estimates, entropy_estimates = flow_sample(sampler_params, rs, num_samples)
        print "Mean loglik:", np.mean(likelihood_estimates.value),\
              "Mean entropy:", np.mean(entropy_estimates.value)
        plot_density(samples.value, "approximating_dist.png")
        return np.mean(likelihood_estimates + entropy_estimates)

    ml_and_grad = value_and_grad(get_batch_marginal_likelihood_estimate)

    # Optimize Langevin parameters.
    def print_ml(ml,params):
        print "log marginal likelihood:", ml
    final_params, final_value = adam(ml_and_grad,sampler_params,num_sampler_optimization_steps,callback=print_ml)


    t1 = time.time()
    print "total runtime", t1-t0


