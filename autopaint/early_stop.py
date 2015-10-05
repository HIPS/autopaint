# Functions to build a sampler based on Langevin dynamics
# that also returns an estimate of the lower bound of the marginal
# likelihood of its output distribution.

import time
import autograd.numpy as np
from autograd import elementwise_grad

from autopaint.util import WeightsParser, \
    entropy_of_a_diagonal_gaussian, entropy_of_a_spherical_gaussian, \
    sum_entropy_lower_bound, exact_log_det, approx_log_det


def gradient_step_track_entropy(gradfun, xs, stepsize, rs, approx):
    """Takes one gradient step, and returns an estimate of the change in entropy."""
    (N, D) = xs.shape
    gradients = gradfun(xs)

    # Hessian-vector product of log-likelihood function.
    # Vectorized version of np.dot(gradfun(xs), vect)).
    hvp = elementwise_grad(lambda xs, vect : np.sum(gradfun(xs) * vect, axis=1))

    def jacobian_vector_product(vect):
        """Product of vect with Jacobian of one step of gradient descent."""
        assert vect.shape == (N,D), vect.shape
        return vect + stepsize * hvp(xs, vect)
    if approx:
        delta_entropy = approx_log_det(jacobian_vector_product, D, N, rs=rs)
    else:
        delta_entropy = exact_log_det(jacobian_vector_product, D, N)
    xs += stepsize * gradients
    return xs, delta_entropy


def gradient_ascent_entropic(gradfun, loglik,entropies, xs, step_size, rs, callback, approx):
    prevL = -np.inf
    num_samples = entropies.shape[0]
    curL = np.mean(loglik(xs)+entropies)
    halfL = np.mean(loglik(xs)[0:.5*num_samples]+entropies[0:.5*num_samples])

    curIter = 0
    #   maxIters = 100
    while halfL > prevL: # and curIter < maxIters:
        if curIter % 10 == 0:
            print 'grad step',curIter
            # print 'cur ent', np.mean(entropies[.5*num_samples:num_samples])
            # print 'cur ll', np.mean(loglik(xs)[.5*num_samples:num_samples])
        if callback: callback(xs=xs, t=t, entropy=delta_entropy)
        new_xs, delta_entropy = gradient_step_track_entropy(gradfun, xs,step_size/np.sqrt(curIter+1), rs, approx=approx)
        # Update entropy estimate.
        new_entropies = delta_entropy+entropies
        prevL = halfL
        new_loglik = loglik(new_xs)
        curL = np.mean(new_loglik+new_entropies)
        halfL = np.mean(loglik(new_xs)[0:.5*num_samples]+new_entropies[0:.5*num_samples])
        if halfL > prevL:
            xs = new_xs
            entropies = new_entropies
        curIter += 1
    print 'difference', np.mean(loglik(xs)[0:.5*num_samples]+entropies[0:.5*num_samples])-np.mean(loglik(xs)[.5*num_samples:num_samples]+entropies[.5*num_samples:num_samples])
    return xs[.5*num_samples:num_samples], entropies[.5*num_samples:num_samples]



def build_early_stop( D, approx):

    # Build parser
    parser = WeightsParser()
    parser.add_shape('mean', D)
    parser.add_shape('log_stddev', D)
    parser.add_shape('log_stepsize', 1)



    def sample_and_run_early_stop(params, loglik_func, rs, num_samples, callback=None):
        num_samples = 2*num_samples
        gradfun = elementwise_grad(loglik_func)
        mean                   = parser.get(params, 'mean')
        stddevs         = np.exp(parser.get(params, 'log_stddev'))
        stepsize       = np.exp(parser.get(params, 'log_stepsize'))

        initial_entropies = np.full(num_samples, entropy_of_a_diagonal_gaussian(stddevs))
        init_xs = mean + rs.randn(num_samples, D) * stddevs
        samples, entropy_estimates = \
            gradient_ascent_entropic(gradfun,loglik = loglik_func, entropies=initial_entropies, xs=init_xs,
                                     step_size=stepsize,
                                     rs=rs, callback=callback, approx=approx)

        loglik_estimates = loglik_func(samples)
        return samples, loglik_estimates, entropy_estimates

    return sample_and_run_early_stop, parser



def build_early_stop_fixed_params( D, approx,mean,log_stddevs,log_stepsize):



    def sample_and_run_early_stop(params, loglik_func, rs, num_samples, callback=None):
        num_samples = 2*num_samples
        gradfun = elementwise_grad(loglik_func)

        stddevs = np.exp(log_stddevs)
        stepsize = np.exp(log_stepsize)
        initial_entropies = np.full(num_samples, entropy_of_a_diagonal_gaussian(stddevs))
        init_xs = mean + rs.randn(num_samples, D) * stddevs

        samples, entropy_estimates = \
            gradient_ascent_entropic(gradfun,loglik = loglik_func, entropies=initial_entropies, xs=init_xs,
                                     step_size=stepsize,
                                     rs=rs, callback=callback, approx=approx)
        loglik_estimates = loglik_func(samples)
        return samples, loglik_estimates, entropy_estimates

    return sample_and_run_early_stop

def build_early_stop_input_params( D, approx,log_stepsize):



    def sample_and_run_early_stop(mean,log_stddevs, loglik_func, rs, num_samples, callback=None):
        num_samples = 2*num_samples
        gradfun = elementwise_grad(loglik_func)

        stddevs = np.exp(log_stddevs)
        stepsize = np.exp(log_stepsize)
        initial_entropies = np.full(num_samples, entropy_of_a_diagonal_gaussian(stddevs))
        init_xs = mean + rs.randn(num_samples, D) * stddevs
        samples, entropy_estimates = \
            gradient_ascent_entropic(gradfun,loglik = loglik_func, entropies=initial_entropies, xs=init_xs,
                                     step_size=stepsize,
                                     rs=rs, callback=callback, approx=approx)

        loglik_estimates = loglik_func(samples)
        return samples, loglik_estimates, entropy_estimates

    return sample_and_run_early_stop