# Functions to build a sampler based on normalizing flows,
# from http://arxiv.org/abs/1505.05770

import autograd.numpy as np
from autograd import elementwise_grad

from .util import WeightsParser, entropy_of_a_diagonal_gaussian

nonlinearity = np.tanh
nonlinearity_grad = elementwise_grad(nonlinearity)

def flow_step(zs, output_weights, transform_weights, bias):
    activations = np.dot(zs, transform_weights) + bias
    zs = zs + np.outer(nonlinearity(activations), output_weights)  # Equation 10
    warp_jacobian = np.dot(output_weights, transform_weights) * nonlinearity_grad(activations)
    entropy_change = np.log(np.abs(1.0 + warp_jacobian))           # Equations 11 and 12
    return zs, entropy_change

def composed_flow(entropies, zs, output_weights, transform_weights, biases, callback):
    num_steps = len(biases)

    for t in xrange(num_steps):
        if callback: callback(zs=zs, t=t, entropy=delta_entropy)
        zs, delta_entropy = flow_step(zs, output_weights[t], transform_weights[t], biases[t])
        entropies += delta_entropy

    return zs, entropies

def build_flow_sampler(loglik_func, D, num_steps):

    parser = WeightsParser()
    parser.add_shape('mean', D)
    parser.add_shape('log_stddev', D)
    parser.add_shape('output weights', (num_steps, D))
    parser.add_shape('transform weights', (num_steps, D))
    parser.add_shape('biases', (num_steps))

    def flow_sample(params, rs, num_samples, callback=None):
        mean = parser.get(params, 'mean')
        stddevs = np.exp(parser.get(params, 'log_stddev'))
        output_weights = parser.get(params, 'output weights')
        transform_weights = parser.get(params, 'transform weights')
        biases = parser.get(params, 'biases')

        initial_entropies = np.full(num_samples, entropy_of_a_diagonal_gaussian(stddevs))
        init_zs = mean + rs.randn(num_samples, D) * stddevs
        samples, entropy_estimates = composed_flow(initial_entropies, init_zs,
                                                   output_weights, transform_weights, biases, callback)
        loglik_estimates = loglik_func(samples)
        return samples, loglik_estimates, entropy_estimates

    return flow_sample, parser
