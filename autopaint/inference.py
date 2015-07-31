import autograd.numpy as np
from autograd import grad

from .util import WeightsParser


def entropy_of_a_gaussian(stddevs):
    # TODO: double check this formula.
    D = len(stddevs)
    return 0.05 * D * (1 + np.log(2*np.pi)) + np.sum(np.log(stddevs))

def entropy_of_a_spherical_gaussian(stddev, D):
    # TODO: double check this formula.
    return 0.05 * D * (1 + np.log(2*np.pi)) + D * np.log(stddev)

def approx_log_det(mvp, D, rs):
    # This should be an unbiased estimator of a lower bound on the log determinant
    # provided the eigenvalues are all greater than 0.317 (solution of
    # log(x) = (x - 1) - (x - 1)**2 = -2 + 3 * x - x**2
    R0 = rs.randn(D) # TODO: Consider normalizing R
    R1 = mvp(R0)
    R2 = mvp(R1)
    return np.dot(R0, -2 * R0 + 3 * R1 - R2)

def exact_log_det(mvp, D):
    mat = np.zeros((D, D))
    eye = np.eye(D)
    for i in range(D):
        mat[:, i] = mvp(eye[:, i])
    return np.log(np.linalg.det(mat))

def gradient_step_track_entropy(gradfun, x, stepsize, rs, approx=False):
    """Takes one gradient step, and returns an estimate of the change in entropy."""
    g = gradfun(x)
    hvp = grad(lambda x, vect : np.dot(gradfun(x), vect)) # Hessian vector product
    jvp = lambda vect : vect + stepsize * hvp(x, vect) # Jacobian vector product
    if approx:
        delta_entropy = approx_log_det(jvp, len(x), rs)
    else:
        delta_entropy = exact_log_det(jvp, len(x))
    x += stepsize * g
    return x, delta_entropy

def sum_entropy_lower_bound(entropy_a, entropy_b, D):
    """Returns lower bound of X + Y given the entropy of X and Y.
    Uses the entropy power inequality.
    https://en.wikipedia.org/wiki/Entropy_power_inequality"""
    return 0.5 * D * np.logaddexp(2.0 * entropy_a / D, 2.0 * entropy_b / D)

def gradient_ascent_entropic(gradfun, entropy, x, stepsizes, noise_sizes, rs, callback, approx=True):
    assert len(stepsizes) == len(noise_sizes)
    D = len(x)
    num_steps = len(stepsizes)

    for t in xrange(num_steps):
        if callback: callback(x=x, t=t, entropy=delta_entropy)
        x, delta_entropy = gradient_step_track_entropy(gradfun, x, stepsizes[t], rs, approx=approx)
        noise = rs.randn(D) * noise_sizes[t]
        x = x + noise

        # Update entropy estimate.
        noise_entropy = entropy_of_a_spherical_gaussian(noise_sizes[t], D)
        entropy = sum_entropy_lower_bound(entropy, noise_entropy, D)

    return x, entropy

def build_langevin_sampler(target_nll_func, D, num_steps):

    # Build parser
    parser = WeightsParser()
    parser.add_shape('mean', D)
    parser.add_shape('stddev', D)
    parser.add_shape('log_stepsizes', num_steps)
    parser.add_shape('log_noise_sizes', num_steps)

    gradfun = grad(target_nll_func)

    def sample_and_run_langevin(params, rs, callback=None):
        mean = parser.get(params, 'mean')
        stddev = parser.get(params, 'stddev')
        stepsizes = np.exp(parser.get(params, 'log_stepsizes'))
        noise_sizes = np.exp(parser.get(params, 'log_noise_sizes'))

        initial_entropy = entropy_of_a_spherical_gaussian(stddev, D)
        init_x = mean + rs.randn(D) * stddev

        z, final_entropy = gradient_ascent_entropic(gradfun, entropy=initial_entropy, x=init_x,
                                                    stepsizes=stepsizes, noise_sizes=noise_sizes,
                                                    rs=rs, callback=callback, approx=True)
        loglik_estimate = -target_nll_func(z)
        marginal_likelihood_estimate = loglik_estimate + final_entropy
        return z, marginal_likelihood_estimate

    return sample_and_run_langevin, parser





