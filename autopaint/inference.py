import autograd.numpy as np
from autograd import grad, elementwise_grad

from .util import WeightsParser, fast_array_from_list,\
    entropy_of_a_diagonal_gaussian, entropy_of_a_spherical_gaussian


def approx_log_det_non_vectorized(mvp, D, rs):
    # This should be an unbiased estimator of a lower bound on the log determinant
    # provided the eigenvalues are all greater than 0.317 (solution of
    # log(x) = (x - 1) - (x - 1)**2 = -2 + 3 * x - x**2
    R0 = rs.randn(D) # TODO: Consider normalizing R
    R1 = mvp(R0)
    R2 = mvp(R1)
    return np.dot(R0, -2 * R0 + 3 * R1 - R2)

def approx_log_det(mvp_vec, D, N, rs):
    # This should be an unbiased estimator of a lower bound on the log determinant
    # provided the eigenvalues are all greater than 0.317 (solution of
    # log(x) = (x - 1) - (x - 1)**2 = -2 + 3 * x - x**2
    # This vectorized version computes N independent bound estimates.
    R0 = rs.randn(N, D) # TODO: Consider normalizing R.
    R1 = mvp_vec(R0)
    R2 = mvp_vec(R1)
    return np.sum(R0 * (-2 * R0 + 3 * R1 - R2), axis=1)  # Row-wise dot products.


def approx_log_det_vectorized_avg(mvp_vec, D, N, num_samples, rs):
    """Averages over several random projections."""
    # Can this be vectorized more directly, without a for loop?
    approx_logdets = [approx_log_det(mvp_vec, D, N, rs) for n in xrange(num_samples)]
    approx_logdets = fast_array_from_list(approx_logdets)
    return np.mean(approx_logdets, axis=0)


def exact_log_det_non_vectorized(jvp, D):
    """mvp is a function that takes in a vector of size D and returns another vector of size D.
    This function builds the Jacobian explicitly, and returns a scalar representing the logdet of the Jacobian."""
    jac = np.zeros((D, D))
    eye = np.eye(D)
    for i in xrange(D):
        jac[:, i] = jvp(eye[:, i])
    sign, logdet = np.linalg.slogdet(jac)
    return logdet


def exact_log_det(mvp_vec, D, N):
    """mvp_vec is a function that takes in a matrix of size N x D and returns another matrix of size N x D.
    This function builds N Jacobians explicitly, and returns a vector representing the logdets of these Jacobians."""
    eye = np.eye(D)
    matlist = []
    for i in xrange(D):
        cur_dir = eye[:, i]
        matlist.append(mvp_vec(np.tile(cur_dir, (N, 1))))
    mat = np.concatenate([np.expand_dims(x, axis=2) for x in matlist], axis=2)
    logdets = []
    for cur_mat in mat:  # Not vectorized, but could be if autograd supported vectorized calls to slogdet.  Does it?
        sign, logdet = np.linalg.slogdet(cur_mat)
        logdets.append(logdet)
    assert len(logdets) == N
    return fast_array_from_list(logdets)


def gradient_step_track_entropy_non_vectorized(gradfun, x, stepsize, rs, approx=False):
    """Takes one gradient step, and returns an estimate of the change in entropy."""
    (N, D) = x.shape
    assert N == 1
    g = gradfun(x)
    hvp = grad(lambda x, vect : np.dot(gradfun(x), vect))  # Hessian-vector product
    jvp = lambda vect : vect + stepsize * hvp(x, vect)     # Jacobian-vector product
    if approx:
        delta_entropy = approx_log_det_non_vectorized(jvp, D, rs)
    else:
        delta_entropy = exact_log_det_non_vectorized(jvp, D)
    x += stepsize * g
    return x, delta_entropy


def gradient_step_track_entropy(gradfun, xs, stepsize, rs, approx):
    """Takes one gradient step, and returns an estimate of the change in entropy."""
    (N, D) = xs.shape
    gradients = gradfun(xs)

    # Hessian-vector product of log-likelihood function.
    # Would use np.dot(gradfun(xs), vect)), but we want to do this in parallel.
    hvp = elementwise_grad(lambda xs, vect : np.sum(gradfun(xs) * vect, axis=1))

    def jacobian_vector_product(vect):
        """Jacobian of one step of gradient descent."""
        assert vect.shape == (N,D), vect.shape
        return vect + stepsize * hvp(xs, vect)
    if approx:
        delta_entropy = approx_log_det(jacobian_vector_product, D, N, rs=rs)
    else:
        delta_entropy = exact_log_det(jacobian_vector_product, D, N)
    xs += stepsize * gradients
    return xs, delta_entropy

def sum_entropy_lower_bound(entropy_a, entropy_b, D):
    """Returns lower bound of X + Y given the entropy of X and Y.
    Uses the entropy power inequality.
    https://en.wikipedia.org/wiki/Entropy_power_inequality"""
    return 0.5 * D * np.logaddexp(2.0 * entropy_a / D, 2.0 * entropy_b / D)

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

    gradfun = elementwise_grad(loglik_func)

    def sample_and_run_langevin(params, rs, num_samples, callback=None):
        mean = parser.get(params, 'mean')
        stddevs = np.exp(parser.get(params, 'log_stddev'))
        stepsizes = np.exp(parser.get(params, 'log_stepsizes'))
        noise_sizes = np.exp(parser.get(params, 'log_noise_sizes'))

        initial_entropies = np.full(num_samples, entropy_of_a_diagonal_gaussian(stddevs))
        init_xs = mean + rs.randn(num_samples, D) * stddevs
        samples, entropy_estimates = gradient_ascent_entropic(gradfun, entropies=initial_entropies, xs=init_xs,
                                                              stepsizes=stepsizes, noise_sizes=noise_sizes,
                                                              rs=rs, callback=callback, approx=approx)

        loglik_estimates = loglik_func(samples)
        return samples, loglik_estimates, entropy_estimates

    return sample_and_run_langevin, parser




