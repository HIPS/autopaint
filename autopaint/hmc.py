# Functions to build a sampler based on Hamiltonian Variational Inference,
# from http://jmlr.org/proceedings/papers/v37/salimans15.pdf

import autograd.numpy as np
from autograd import elementwise_grad

from .util import WeightsParser, build_logprob_mvn


def simple_linear_model(ll_grad, A, B, stddevs, rs):
    def sample(zs):
        (N, D) = zs.shape
        noise = rs.randn(N, D)
        samples = np.dot(zs, A) + np.dot(ll_grad(zs), B) + noise * stddevs
        return samples

    def log_prob(vs, zs):
        mvn = build_logprob_mvn(np.dot(zs, A) + np.dot(ll_grad(zs), B), np.diag(stddevs ** 2), pseudo_inv=False)
        return mvn(vs)

    return sample, log_prob


def hamiltonian_dynamics(zs, vs, log_like_grad, hmc_stepsize, mass_mat, leap_steps):
    for i in xrange(leap_steps):
        vs = vs + hmc_stepsize * log_like_grad(zs) / 2.0  # Half step in momentum.
        zs = zs + hmc_stepsize * np.dot(vs, mass_mat)     # Full sample step.
        vs = vs + hmc_stepsize * log_like_grad(zs) / 2.0  # Half step in momentum
    return zs, vs


def run_hmc(init_zs, log_like, loglik_func_grad, hmc_stepsize, mass_mat, v_A, v_B, v_cov, rev_A, rev_B, rev_cov,
            num_iters, leap_steps, rs, callback):
    # Generate q and r models
    q_sample, q_log_prob = simple_linear_model(loglik_func_grad, v_A, v_B, v_cov, rs)
    r_sample, r_log_prob = simple_linear_model(loglik_func_grad, rev_A, rev_B, rev_cov, rs)
    (N, D) = init_zs.shape
    L = np.zeros(N)
    zs = init_zs
    for t in xrange(num_iters):
        #Draw momentum
        vs = q_sample(zs)
        #Apply hamiltonian dynamics with current samples
        new_zs, new_vs = hamiltonian_dynamics(zs, vs, loglik_func_grad, hmc_stepsize, mass_mat, leap_steps)
        #Compute alpha
        log_rev_prob_new_v = r_log_prob(new_vs, new_zs)
        log_prob_v = q_log_prob(vs, zs)
        log_alpha = log_like(new_zs) + log_rev_prob_new_v - log_like(zs) - log_prob_v
        L = L + log_alpha
        zs = new_zs
    return zs, L


def build_hmc_sampler(loglik_func, D, num_steps, leap_steps):
    # Create initialization parameters
    parser = WeightsParser()
    parser.add_shape('mean', D)
    parser.add_shape('log_stddev', D)

    #Create HMC parameters
    parser.add_shape('hmc_log_stepsize', 1)
    parser.add_shape('mass_mat', (D, D))

    #Momentum initialization parameters
    #Diagonal multivariate gaussian
    parser.add_shape('v_A', (D, D))
    parser.add_shape('v_B', (D, D))
    parser.add_shape('v_log_cov', D)

    #Create reverse model parameters
    #Simple reverse model (diagonal multivariate gaussian)
    parser.add_shape('rev_A', (D, D))
    parser.add_shape('rev_B', (D, D))
    parser.add_shape('rev_log_cov', D)

    loglik_func_grad = elementwise_grad(loglik_func)

    def hmc_sample(params, rs, num_samples, callback=None):
        """Generate a samples from HMC with given parameters and return them,
           as well as an unbiased estimate on the lower bound."""

        # Unpack parameters
        mean = parser.get(params, 'mean')
        stddevs = np.exp(parser.get(params, 'log_stddev'))
        hmc_stepsize = np.exp(parser.get(params, 'hmc_log_stepsize'))
        mass_mat = parser.get(params, 'mass_mat')
        v_A = parser.get(params, 'v_A')
        v_B = parser.get(params, 'v_B')
        v_cov = np.exp(parser.get(params, 'v_log_cov'))
        rev_A = parser.get(params, 'rev_A')
        rev_B = parser.get(params, 'rev_B')
        rev_cov = np.exp(parser.get(params, 'rev_log_cov'))

        #Create initial sample and combine its log_lik and its entropy
        init_zs = mean + rs.randn(num_samples, D) * stddevs
        init_ll = loglik_func(init_zs)
        init_log_prob_mvn = build_logprob_mvn(mean, np.diag(stddevs ** 2), pseudo_inv=False)
        init_ent = init_log_prob_mvn(init_zs)
        init_L_est = init_ll - init_ent

        samples, lower_bound_est = run_hmc(init_zs, loglik_func, loglik_func_grad, hmc_stepsize, mass_mat, v_A, v_B,
                                           v_cov, rev_A, rev_B, rev_cov, num_steps, leap_steps, rs, callback)
        return samples, lower_bound_est + init_L_est

    return hmc_sample, parser
