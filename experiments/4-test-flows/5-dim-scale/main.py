# Main demo script
import autograd.numpy as np
from autograd import value_and_grad
import numpy.linalg
import time

from autopaint.flows import build_flow_sampler
from autopaint.plotting import plot_density,plot_line
from autopaint.util import build_logprob_mvn, log_inv_rosenbrock
from autopaint.optimizers import sga,adam

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
    num_sampler_optimization_steps = 1
    sampler_learn_rate = 1e-2


    D = 2
    maxD = 200
    scale_list = []
    d_list = []
    while D < maxD:
        init_mean = np.zeros(D)
        init_log_stddevs = np.log(0.1*np.ones(D))
        init_output_weights = 0.1*rs.randn(num_steps, D)
        init_transform_weights = 0.1*rs.randn(num_steps, D)
        init_biases = 0.1*rs.randn(num_steps)

        #logprob_mvn = build_logprob_mvn(mean=np.array([0.2,0.4]), cov=np.array([[1.0,0.9], [0.9,1.0]]))
        flow_sample, parser = build_flow_sampler(log_inv_rosenbrock, D, num_steps)

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

        startOpt = time.time()
        final_params, final_value = adam(ml_and_grad,sampler_params,num_sampler_optimization_steps,callback=print_ml)
        endOpt = time.time()

        scale_list.append(endOpt-startOpt)
        d_list.append(D)
        #Update D
        D = D+10

    y = np.asarray(scale_list)
    x = np.asarray(d_list)
    plot_line(x,y,'scale_dim.png')


    t1 = time.time()
    print "total runtime", t1-t0


