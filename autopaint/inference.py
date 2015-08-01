import autograd.numpy as np
from autograd import grad, elementwise_grad

from .util import WeightsParser, fast_array_from_list


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
    for i in xrange(D):
        mat[:, i] = mvp(eye[:, i])
    sign, logdet = np.linalg.slogdet(mat)
    return logdet

def exact_log_det_vectorized(mvp_vec, D, N):
    mat = np.zeros((N, D, D))
    eye = np.eye(D)
    for i in xrange(D):
        mat[:, :, i] =  mvp_vec(eye[:, i])
    logdets = []
    for cur_mat in mat:  # Not vectorized, but could be if autograd supported vectorized calls to slogdet.
        sign, logdet = np.linalg.slogdet(cur_mat)
        logdets.append(logdet)
    return fast_array_from_list(logdets)

def gradient_step_track_entropy(gradfun, init_x, stepsize, rs, approx=False):
    """Takes one gradient step, and returns an estimate of the change in entropy."""
    x = init_x
    (N, D) = x.shape
    assert N == 1
    g = gradfun(x)
    hvp = grad(lambda x, vect : np.dot(gradfun(x), vect))  # Hessian-vector product
    jvp = lambda vect : vect + stepsize * hvp(x, vect)     # Jacobian-vector product
    if approx:
        delta_entropy = approx_log_det(jvp, D, rs)
    else:
        delta_entropy = exact_log_det(jvp, D)
    x += stepsize * g
    return x, delta_entropy

def gradient_step_track_entropy_vectorized(gradfun, init_xs, stepsize, rs, approx=False):
    """Takes one gradient step, and returns an estimate of the change in entropy."""
    xs = init_xs
    (N, D) = xs.shape
    gs = gradfun(xs)
    hvp = elementwise_grad(lambda xs, vect : np.dot(gradfun(xs), vect))  # Hessian-vector product
    jvp = lambda vect : vect + stepsize * hvp(xs, vect)     # Jacobian-vector product
    if approx:
        delta_entropy = approx_log_det(jvp, D, rs)
    else:
        delta_entropy = exact_log_det_vectorized(jvp, D, N)
    xs += stepsize * gs
    return xs, delta_entropy

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
        entropy += delta_entropy
        noise_entropy = entropy_of_a_spherical_gaussian(noise_sizes[t], D)
        entropy = sum_entropy_lower_bound(entropy, noise_entropy, D)

    return x, entropy

def build_langevin_sampler(loglik_func, D, num_steps):

    # Build parser
    parser = WeightsParser()
    parser.add_shape('mean', D)
    parser.add_shape('stddev', D)
    parser.add_shape('log_stepsizes', num_steps)
    parser.add_shape('log_noise_sizes', num_steps)

    gradfun = grad(loglik_func)

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
        loglik_estimate = loglik_func(z)
        marginal_likelihood_estimate = loglik_estimate + final_entropy
        return z, marginal_likelihood_estimate

    return sample_and_run_langevin, parser





