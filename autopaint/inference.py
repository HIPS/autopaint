# Functions to build a sampler based on Langevin dynamics
# that also returns an estimate of the lower bound of the marginal
# likelihood of its output distribution.

import autograd.numpy as np
from autograd import elementwise_grad

from .util import WeightsParser, sigmoid,\
    entropy_of_a_diagonal_gaussian, entropy_of_a_spherical_gaussian, \
    sum_entropy_lower_bound, exact_log_det, approx_log_det


def gradient_step_track_entropy(gradfun, xs, stepsize, rs, approx):
    """Takes one gradient step, and returns an estimate of the change in entropy."""
    (N, D) = xs.shape
    gradients = gradfun(xs)

    # Hessian-vector product of log-likelihood function.
    # Would use np.dot(gradfun(xs), vect)), but we want to do this in parallel.
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


def gradient_ascent_entropic(gradfun, entropies, xs, stepsizes, noise_sizes, rs, callback, approx):
    assert len(stepsizes) == len(noise_sizes)
    (N, D) = xs.shape
    num_steps = len(stepsizes)

    for t in xrange(num_steps):
        if callback: callback(xs=xs, t=t, entropy=delta_entropy)
        xs, delta_entropy = gradient_step_track_entropy(gradfun, xs, stepsizes[t], rs, approx=approx)
        noise = rs.randn(N, D) * noise_sizes[t]
        xs = xs + noise

        # Update entropy estimate.
        entropies += delta_entropy
        noise_entropies = entropy_of_a_spherical_gaussian(noise_sizes[t], D)
        entropies = sum_entropy_lower_bound(entropies, noise_entropies, D)

    return xs, entropies


def build_langevin_sampler(loglik_func, D, num_steps, approx):

    # Build parser
    parser = WeightsParser()
    parser.add_shape('mean', D)
    parser.add_shape('log_stddev', D)
    parser.add_shape('log_stepsizes', num_steps)
    parser.add_shape('log_noise_sizes', num_steps)
    parser.add_shape('log_gradient_scales', D)
    parser.add_shape('invsig_gradient_power', 1)

    gradfun = elementwise_grad(loglik_func)

    def sample_and_run_langevin(params, rs, num_samples, callback=None):
        mean = parser.get(params, 'mean')
        stddevs = np.exp(parser.get(params, 'log_stddev'))
        stepsizes = np.exp(parser.get(params, 'log_stepsizes'))
        noise_sizes = np.exp(parser.get(params, 'log_noise_sizes'))
        gradient_scales = np.exp(parser.get(params, 'log_gradient_scales'))
        gradient_power = sigmoid(parser.get(params, 'invsig_gradient_power'))

        def scaled_gradfun(x):
            # Todo:  Think about whether this should go inside gradfun,
            # or in the step-size calculation. (and re-check bounds).
            # Should we also scale the gradient before we apply the power?
            gradient = gradfun(x)
            gradient_magnitude = np.sqrt(np.sum(gradient**2, 1, keepdims=1))
            return gradient_scales * gradient * gradient_magnitude**(gradient_power - 1.0)

        initial_entropies = np.full(num_samples, entropy_of_a_diagonal_gaussian(stddevs))
        init_xs = mean + rs.randn(num_samples, D) * stddevs
        samples, entropy_estimates = \
            gradient_ascent_entropic(scaled_gradfun, entropies=initial_entropies, xs=init_xs,
                                     stepsizes=stepsizes, noise_sizes=noise_sizes,
                                     rs=rs, callback=callback, approx=approx)

        loglik_estimates = loglik_func(samples)
        return samples, loglik_estimates, entropy_estimates

    return sample_and_run_langevin, parser
