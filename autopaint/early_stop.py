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
    curL = loglik(xs)+entropies
    curIter = 0
    while curL > prevL:
        if callback: callback(xs=xs, t=t, entropy=delta_entropy)
        new_xs, delta_entropy = gradient_step_track_entropy(gradfun, xs,step_size, rs, approx=approx)
        # Update entropy estimate.
        new_entropies = delta_entropy+entropies
        prevL = curL
        new_loglik = loglik(new_xs)
        curL = new_loglik+new_entropies
        if curL > prevL:
            xs = new_xs
            entropies = new_entropies
        curIter += 1
    return xs, entropies

def build_early_stop( D, approx):

    # Build parser
    parser = WeightsParser()
    parser.add_shape('mean', D)
    parser.add_shape('log_stddev', D)
    parser.add_shape('log_stepsize', 1)



    def sample_and_run_early_stop(params, loglik_func, rs, num_samples, callback=None):
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
