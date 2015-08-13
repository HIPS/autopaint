
# Functions to build a sampler based on normalizing flows,
# from http://arxiv.org/abs/1505.05770

import autograd.numpy as np
from autograd import grad
from autograd import elementwise_grad

from .util import WeightsParser, build_logprob_mvn

def simple_linear_model(ll_grad,A,B,stddevs):
    def sample(zs):
        (N,D) = zs.shape
        noise = np.randn(N,D)
        samples = np.dot(A,zs)+np.dot(B,ll_grad(zs))+noise*stddevs
        return samples
    def log_prob(vs,zs):
        mvn = build_logprob_mvn(np.dot(A,zs)+np.dot(B,ll_grad(zs)), np.diag(stddevs))
        return mvn(vs)
    return sample,log_prob

def hamiltonian_dynamics(zs,vs,log_like_grad,hmc_stepsize,mass_mat,leap_steps):
    for i in xrange(leap_steps):
        #Half step in momentum
        vs = vs + hmc_stepsize*log_like_grad(zs)/2
        #Full sample step
        zs = zs + hmc_stepsize*vs
        #Half step in momentum
        vs = vs + hmc_stepsize*log_like_grad(zs)/2
    return zs,vs


def run_hmc(init_zs,log_like,loglik_func_grad,hmc_stepsize, mass_mat, v_A,v_B,v_cov,rev_A,rev_B, rev_cov, num_iters, callback):
    #Generate q and r models
    q_sample, q_log_prob = simple_linear_model(loglik_func_grad,v_A,v_B,v_cov)
    r_sample, r_log_prob = simple_linear_model(loglik_func_grad,rev_A,rev_B,rev_cov)
    L = 0
    zs = init_zs
    for t in xrange(num_iters):
        #Draw momentum
        vs = q_sample(zs)
        #Apply hamiltonian dynamics with current sample, compute new samples and delta entropies
        new_zs, new_vs = hamiltonian_dynamics(zs,vs,hmc_stepsize,mass_mat)
        #Compute alpha
        log_rev_prob_new_v = r_log_prob(new_vs,new_zs)
        log_prob_v = q_log_prob(vs,zs)
        log_alpha = log_like(new_zs)+log_rev_prob_new_v-log_like(zs)-log_prob_v
        L = L + log_alpha
        zs = new_zs
    #Return lower bound estimates
    return zs,L


def build_hmc_sampler(loglik_func, D, num_steps):

    #Create params
    #Create initialization parameters
    parser = WeightsParser()
    parser.add_shape('mean', D)
    parser.add_shape('log_stddev', D)

    #Create HMC parameters
    parser.add_shape('hmc_stepsize',1)
    parser.add_shape('mass_mat',(D,D))

    #Momentum initialization parameters
    #Diagonal multivariate gaussian
    parser.add_shape('v_A',(D,D))
    parser.add_shape('v_B'(D,D))
    parser.add_shape('v_cov',D)

    #Create reverse model parameters
    #Simple reverse model (diagonal multivariate gaussian)
    parser.add_shape('rev_A',D)
    parser.add_shape('rev_B',D)
    parser.add_shape('rev_cov',D)

    loglik_func_grad = grad(loglik_func)

    def hmc_sample(params, loglik_func_grad, rs, num_samples, callback=None):
        #Generate a samples from HMC with given parameters and return them, as well as an unbiased estimate on the lower bound

        #Unparse parameters
        mean = parser.get(params, 'mean')
        stddevs = np.exp(parser.get(params, 'log_stddev'))
        hmc_stepsize = parser.get(params, 'hmc_stepsize')
        mass_mat = parser.get(params, 'mass_mat')
        v_A = parser.get(params, 'v_A')
        v_B = parser.get(params,'v_B')
        v_cov = parser.get(params, 'v_cov')
        rev_A = parser.get(params,'rev_A')
        rev_B = parser.get(params,'rev_B')
        rev_cov = parser.get(params,'rev_cov')

        #Create initial sample and combine its log_lik and its entropy
        init_zs = mean + rs.randn(num_samples, D) * stddevs
        init_ll = loglik_func(init_zs)
        init_log_prob_mvn = build_logprob_mvn(mean,np.diag(stddevs**2))
        init_ent = init_log_prob_mvn(init_zs)
        init_L_est = init_ll-init_ent

        #Get samples with lower_bound estimate
        samples, lower_bound_est = run_hmc(init_zs,loglik_func,loglik_func_grad,hmc_stepsize,mass_mat,v_A,v_B,v_cov,rev_A,rev_B,rev_cov, callback)

        lower_bound_est = lower_bound_est + init_L_est
        #Return samples, lower_bound_est
        return samples, lower_bound_est


    return hmc_sample, parser
