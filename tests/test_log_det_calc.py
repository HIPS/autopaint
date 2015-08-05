import autograd.numpy as np
import autograd.numpy.random as npr

from autopaint.util import fast_array_from_list
from autopaint.inference import exact_log_det, exact_log_det_vectorized, approx_log_det,\
    gradient_step_track_entropy_vectorized, gradient_step_track_entropy, approx_log_det_vectorized,\
    build_langevin_sampler, sum_entropy_lower_bound
from autopaint.util import logprob_two_moons, logprob_mvn

from autograd import grad, elementwise_grad
from autograd.util import quick_grad_check, check_grads

def test_exact_log_det():
    D = 10
    rs = npr.RandomState(0)
    mat = np.eye(D) - 0.1 * np.diag(rs.rand(D))
    mvp = lambda v : np.dot(mat, v)
    assert exact_log_det(mvp, D) == np.log(np.linalg.det(mat))

def test_exact_log_det_vectorized():
    D = 10
    N = 7
    rs = npr.RandomState(0)
    mats = []
    exact_logdets = []
    for i in xrange(N):
        # Build N different functions, each multiplying against a different matrix.
        cur_mat = np.eye(D) - 0.1 * np.diag(rs.rand(D))
        mats.append(cur_mat)
        cur_func = lambda v : np.dot(cur_mat, v)
        exact_logdets.append(exact_log_det(cur_func, D))

    def mvp_vec(v):
        """Vectorized version takes in N vectors of length D, and multiples
        each v against the corresponding matrix in the list."""
        assert v.shape == (N, D), v.shape
        mvps = []
        for i in xrange(N):
            print "i:", i, v[i]
            mvps.append(np.dot(mats[i], v[i]))
        return fast_array_from_list(mvps)
    vec_logdets = exact_log_det_vectorized(mvp_vec, D, N)

    assert np.all(vec_logdets == exact_logdets),\
        "vectorized: {} non-vectorized: {}".format(vec_logdets, exact_logdets)


def test_approx_log_det_vectorized():
    D = 10
    N = 7
    rs = npr.RandomState(0)
    rs2 = npr.RandomState(0)
    mats = []
    alds = []
    for i in xrange(N):
        cur_mat = np.eye(D) - 0.1 * np.diag(rs.rand(D))
        mats.append(cur_mat)
        cur_func = lambda v : np.dot(cur_mat, v.T)
        alds.append(approx_log_det(cur_func, D, rs=rs2))
    alds = np.array(alds)

    def mvp_vec(v):
        """Vectorized version takes in N vectors of length D, and multiples
           each v against the corresponding matrix in the list.
           Takes in a matrix of length N x D, returns an N x D matrix."""
        assert v.shape == (N, D), v.shape
        mvps = []
        for i in xrange(N):
            mvps.append(np.dot(mats[i], v[i]))
        retval = fast_array_from_list(mvps)
        assert retval.shape == (N,D), retval.shape
        return fast_array_from_list(mvps)

    vec_logdets = approx_log_det_vectorized(mvp_vec, D, N, rs=npr.RandomState(0))

    assert np.all(vec_logdets - alds < 0.0001), "vectorized: {} non-vectorized: {}, diff: {}".format(vec_logdets, alds, vec_logdets - alds)


def test_entropy_bound_vectorized_vs_not():
    D = 2
    N = 1
    rs = npr.RandomState(0)
    stepsize = 0.1
    approx = False

    xs = rs.randn(N,D)

    gradfun = grad(logprob_two_moons)
    gradfun_vec = elementwise_grad(logprob_two_moons)

    new_xs = []
    new_es = []
    for i in xrange(N):
        cur_x = np.reshape(xs[i], (1, D))
        cur_new_x, cur_new_e = gradient_step_track_entropy(gradfun, x=cur_x.copy(), stepsize=stepsize, rs=None, approx=approx)
        new_xs.append(cur_new_x)
        new_es.append(cur_new_e)

    vec_new_xs, vec_new_es = gradient_step_track_entropy_vectorized(gradfun_vec, xs=xs.copy(), stepsize=stepsize, rs=None, approx=approx)

    for i in xrange(N):
        assert np.all(vec_new_xs[i] == new_xs[i]), "vectorized: {} non-vectorized: {}".format(vec_new_xs[i], new_xs[i])
    assert np.all(vec_new_es[i] == new_es[i]),     "vectorized: {} non-vectorized: {}".format(vec_new_es, new_es)


def test_approx_log_det():
    D = 100
    rs = npr.RandomState(0)
    mat = np.eye(D) - 0.1 * np.diag(rs.rand(D))
    mvp = lambda v : np.dot(mat, v)
    N_trials = 10000
    approx = 0
    for i in xrange(N_trials):
        approx += approx_log_det(mvp, D, rs)
    approx = approx / N_trials
    exact = exact_log_det(mvp, D)
    assert exact > approx > (exact - 0.1 * np.abs(exact))
    print exact, approx


def test_likelihood_gradient():
    quick_grad_check(logprob_two_moons, np.atleast_2d(np.array([0.1, 0.2]).T))

def test_likelihood_gradient():
    quick_grad_check(logprob_two_moons, np.atleast_2d(np.array([0.1, 0.2]).T))


def test_meta_gradient_with_langevin():
    num_samples = 4
    num_langevin_steps = 3

    D = 2
    init_mean = npr.randn(D) * 0.01
    init_log_stddevs = np.log(1*np.ones(D)) + npr.randn(D) * 0.01
    init_log_stepsizes = np.log(0.01*np.ones(num_langevin_steps)) + npr.randn(num_langevin_steps) * 0.01
    init_log_noise_sizes = np.log(.001*np.ones(num_langevin_steps)) + npr.randn(num_langevin_steps) * 0.01

    sample_and_run_langevin, parser = build_langevin_sampler(logprob_two_moons, D, num_langevin_steps, approx=False)

    sampler_params = np.zeros(len(parser))
    parser.put(sampler_params, 'mean', init_mean)
    parser.put(sampler_params, 'log_stddev', init_log_stddevs)
    parser.put(sampler_params, 'log_stepsizes', init_log_stepsizes)
    parser.put(sampler_params, 'log_noise_sizes', init_log_noise_sizes)

    def get_batch_marginal_likelihood_estimate(sampler_params):
        rs = np.random.npr.RandomState(0)
        samples, loglik_estimates, entropy_estimates = sample_and_run_langevin(sampler_params, rs, num_samples)
        marginal_likelihood_estimates = loglik_estimates + entropy_estimates
        return np.mean(marginal_likelihood_estimates)

    check_grads(get_batch_marginal_likelihood_estimate, sampler_params)


def test_meta_gradient_with_langevin_mvn():
    num_samples = 4
    num_langevin_steps = 3

    D = 2
    init_mean = npr.randn(D) * 0.01
    init_log_stddevs = np.log(1*np.ones(D)) + npr.randn(D) * 0.01
    init_log_stepsizes = np.log(0.01*np.ones(num_langevin_steps)) + npr.randn(num_langevin_steps) * 0.01
    init_log_noise_sizes = np.log(.001*np.ones(num_langevin_steps)) + npr.randn(num_langevin_steps) * 0.01

    sample_and_run_langevin, parser = build_langevin_sampler(logprob_mvn, D, num_langevin_steps, approx=False)

    sampler_params = np.zeros(len(parser))
    parser.put(sampler_params, 'mean', init_mean)
    parser.put(sampler_params, 'log_stddev', init_log_stddevs)
    parser.put(sampler_params, 'log_stepsizes', init_log_stepsizes)
    parser.put(sampler_params, 'log_noise_sizes', init_log_noise_sizes)

    def get_batch_marginal_likelihood_estimate(sampler_params):
        rs = np.random.npr.RandomState(0)
        samples, loglik_estimates, entropy_estimates = sample_and_run_langevin(sampler_params, rs, num_samples)
        marginal_likelihood_estimates = loglik_estimates + entropy_estimates
        return np.mean(marginal_likelihood_estimates)

    check_grads(get_batch_marginal_likelihood_estimate, sampler_params)


def test_entropy_lower_bound():
    assert sum_entropy_lower_bound(1.0, 0.0) == 1.0