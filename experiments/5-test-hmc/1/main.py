# Main demo script
import autograd.numpy as np
from autograd import value_and_grad
from autograd.util import check_grads
import numpy.linalg
import time

from autopaint.hmc import build_hmc_sampler
from autopaint.plotting import plot_density
from autopaint.util import build_logprob_mvn


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

    num_samples = 100
    num_steps = 20
    leap_steps = 20
    num_sampler_optimization_steps = 400
    sampler_learn_rate = 1e-6

    D = 2
    init_mean = np.zeros(D)
    init_log_stddevs = np.log(.01*np.ones(D))
    hmc_log_stepsize = np.log(.1)
    mass_mat = np.eye(D)
    v_A = np.zeros(D)
    v_B = np.zeros(D)
    v_log_cov = np.log(.01*np.ones(D))
    rev_A = np.zeros(D)
    rev_B = np.zeros(D)
    rev_log_cov = np.log(.01*np.ones(D))

    # logprob_mvn = build_logprob_mvn(mean=np.array([0.0,0.0]), cov=np.array([[1.0,0.9], [0.9,1.0]]),pseudo_inv = False)
    hmc_sample, parser = build_hmc_sampler(logprob_two_moons, D, num_steps,leap_steps)

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

    def get_batch_marginal_likelihood_estimate(sampler_params):
        samples, L_ests = hmc_sample(sampler_params, rs, num_samples,leap_steps)
        plot_density(samples.value, "approximating_dist.png")
        print 'empirical mean', np.mean(samples,axis=0).value
        print 'empirical cov', np.cov((samples.T).value)
        return np.mean(L_ests)

    ml_and_grad = value_and_grad(get_batch_marginal_likelihood_estimate)
    # Optimize Langevin parameters.
    for i in xrange(num_sampler_optimization_steps):
        ml, dml = ml_and_grad(sampler_params)
        print "log marginal likelihood:", ml
        # print 'grad magn', np.linalg.norm(dml)
        sampler_params = sampler_params + sampler_learn_rate * dml

    t1 = time.time()
    print "total runtime", t1-t0


