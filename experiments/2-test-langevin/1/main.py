# Main demo script
import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
import numpy.linalg
import matplotlib.image
import pickle

from autopaint.plotting import *
from autopaint.recognition import build_langevin_sampler

def test_energy_two_moons(z):
    z1 = z[0]
    z2 = z[1]
    return 0.5 * ((np.sqrt(z1**2 + z2**2) - 2 ) / 0.4)**2\
            - np.logaddexp(-0.5 * ((z1 - 2) / 0.6)**2, -0.5 * ((z1 + 2) / 0.6)**2)

def test_energy_wiggle(z):
    z1 = z[0]
    z2 = z[1]
    return 0.5 * (z2 - np.sin(2.0 * np.pi * z1 / 4.0) / 0.4 )**2 + 0.2 * (z1**2 + z2**2)



if __name__ == '__main__':

    num_samples = 100
    num_langevin_steps = 5

    D = 2
    init_mean = np.zeros(D)
    init_stddevs = np.ones(D)
    init_stepsizes = 0.1*np.ones(D)   # TODO: change to log
    init_noise_sizes = 0.1*np.ones(D) # TODO: change to log

    rs = np.random.npr.RandomState(0)

    sample_and_run_langevin, parser = build_langevin_sampler(test_energy_wiggle, D)

    sampler_params = np.zeros(len(parser))
    parser.put(sampler_params, 'mean', init_mean)
    parser.put(sampler_params, 'stddev', init_stddevs)
    parser.put(sampler_params, 'stepsizes', init_stepsizes)
    parser.put(sampler_params, 'noise_sizes', init_noise_sizes)

    samples = np.zeros((num_samples, D))
    for s in range(num_samples):
        z, marginal_likelihood_estimate = sample_and_run_langevin(sampler_params, rs)
        samples[s, :] = z

    plot_density(samples, "approximating_dist.png")


