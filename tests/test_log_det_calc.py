import autograd.numpy as np
import autograd.numpy.random as npr

from autopaint.util import fast_array_from_list
from autopaint.inference import exact_log_det, exact_log_det_vectorized, approx_log_det,\
    gradient_step_track_entropy_vectorized, gradient_step_track_entropy

from autograd import grad, elementwise_grad

def test_exact_log_det():
    D = 100
    rs = npr.RandomState(0)
    mat = np.eye(D) - 0.1 * np.diag(rs.rand(D))
    mvp = lambda v : np.dot(mat, v)
    assert exact_log_det(mvp, D) == np.log(np.linalg.det(mat))

def test_exact_log_det_vectorized():
    D = 100
    N = 7
    rs = npr.RandomState(0)
    mats = []
    mvps = []
    for i in xrange(N):
        cur_mat = np.eye(D) - 0.1 * np.diag(rs.rand(D))
        mats.append(cur_mat)
        mvps.append(lambda v : np.dot(cur_mat, v))

    def mvp_vec(v):
        # Vectorized version multiplied v against each matrix in our list.
        mvps = []
        for i in xrange(N):
            mvps.append(np.dot(cur_mat, v))
        return fast_array_from_list(mvps)

    vec_logdets = exact_log_det_vectorized(mvp_vec, D, N)

    for i in xrange(N):
        assert vec_logdets[i] == exact_log_det(mvps[i], D), "vld: {} old fashioned: {}".format(vec_logdets[i], exact_log_det(mvps[i], D))

def logprob_two_moons(z):
    z1 = z[:, 0]
    z2 = z[:, 1]
    print z
    return - 0.5 * ((np.sqrt(z1**2 + z2**2) - 2 ) / 0.4)**2\
            + np.logaddexp(-0.5 * ((z1 - 2) / 0.6)**2, -0.5 * ((z1 + 2) / 0.6)**2)

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
        cur_new_x, cur_new_e = gradient_step_track_entropy(gradfun, init_x=cur_x.copy(), stepsize=stepsize, rs=None, approx=approx)
        new_xs.append(cur_new_x)
        new_es.append(cur_new_e)

    vec_new_xs, vec_new_es = gradient_step_track_entropy_vectorized(gradfun_vec, init_xs=xs.copy(), stepsize=stepsize, rs=None, approx=approx)
    vec_new_xs, vec_new_es = gradient_step_track_entropy_vectorized(gradfun_vec, init_xs=xs.copy(), stepsize=stepsize, rs=None, approx=approx)
    vec_new_xs, vec_new_es = gradient_step_track_entropy_vectorized(gradfun_vec, init_xs=xs.copy(), stepsize=stepsize, rs=None, approx=approx)

    for i in xrange(N):
        assert np.all(vec_new_xs[i] == new_xs[i]), "vectorized: {} non-vectorized: {}".format(vec_new_xs[i], new_xs[i])

    assert np.all(vec_new_es[i] == new_es[i]),     "vectorized: {} non-vectorized: {}".format(vec_new_es, new_es)

test_entropy_bound_vectorized_vs_not()

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
