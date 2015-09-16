# Main demo script
import pickle
import autograd.numpy as np
from autograd import value_and_grad,grad
import numpy.linalg
import time
import matplotlib.pyplot as plt

from autopaint.optimizers import adam
from autopaint.flows import build_flow_sampler
from autopaint.plotting import plot_density
from autopaint.util import build_logprob_mvn,create_banded_cov
from autopaint.neuralnet import make_classification_nn
from autopaint.plotting import plot_images

cov = np.array([[1.0, 0.9], [0.9, 1.0]])
pinv = np.linalg.pinv(cov)
(sign, logdet) = numpy.linalg.slogdet(cov)
const =  -0.5 * 2 * np.log(2*np.pi) - 0.5 * logdet
def logprob_mvn(z):
    return const - 0.5 * np.dot(np.dot(z.T, pinv), z)

param_scale = 0.1
samples_per_image = 1
latent_dimensions = 10
hidden_units = 500


if __name__ == '__main__':

    t0 = time.time()
    rs = np.random.npr.RandomState(0)

    # load_and_pickle_binary_mnist()
    with open('../../../autopaint/mnist_binary_data.pkl') as f:
        N_data, train_images, train_labels, test_images, test_labels = pickle.load(f)

   # Create aevb function
    # Training parameters

    D = train_images.shape[1]

    # run_aevb(train_images)
    # with open('parameters.pkl') as f:
    #     parameters = pickle.load(f)

    with open('mnist_models.pkl') as f:
        trained_weights, all_mean, all_cov = pickle.load(f)

    banded_cov = create_banded_cov(all_cov.shape[0],100)
    # Build likelihood model.
    L2_reg = 1
    layer_sizes = [784, 200, 100, 10]
    num_weights, make_predictions, likelihood = make_classification_nn(layer_sizes)
    classifier_loglik = lambda image, c: make_predictions(trained_weights, np.atleast_2d(image))[:, c]

    image_prior = build_logprob_mvn(all_mean, banded_cov)
    # Combine prior and likelihood.
    model_ll = lambda image, c: image_prior(image) +classifier_loglik(image, c)

    num_samples = 100
    num_steps = 32
    num_sampler_optimization_steps = 400
    sampler_learn_rate = 0.01

    D = 784
    init_mean = np.zeros(D)
    init_log_stddevs = np.log(0.1*np.ones(D))
    init_output_weights = 0.1*rs.randn(num_steps, D)
    init_transform_weights = 0.1*rs.randn(num_steps, D)
    init_biases = 0.1*rs.randn(num_steps)

    cur_class = 9
    def class_ll(image):
        return model_ll(image,cur_class)
    #logprob_mvn = build_logprob_mvn(mean=np.array([0.2,0.4]), cov=np.array([[1.0,0.9], [0.9,1.0]]))
    flow_sample, parser = build_flow_sampler(D, num_steps)
    parser.add_shape('mean',  D)
    parser.add_shape('log_stddevs', D)

    sampler_params = np.zeros(len(parser))
    parser.put(sampler_params, 'mean', init_mean)
    parser.put(sampler_params, 'log_stddev', init_log_stddevs)
    parser.put(sampler_params, 'output weights', init_output_weights)
    parser.put(sampler_params, 'transform weights', init_transform_weights)
    parser.put(sampler_params, 'biases', init_biases)

    def get_batch_marginal_likelihood_estimate(sampler_params,i):
        samples, entropy_estimates = flow_sample(sampler_params,num_samples, rs)
        likelihood_estimates = class_ll(samples)
        print "Mean loglik:", np.mean(likelihood_estimates.value),\
              "Mean entropy:", np.mean(entropy_estimates.value)

        images_per_row = 10
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
        plot_images(samples.value, ax, ims_per_row=images_per_row)
        plt.savefig('samples.png')
        return np.mean(likelihood_estimates + entropy_estimates)

    grad_func = grad(get_batch_marginal_likelihood_estimate)

    # Optimize Langevin parameters.
    adam(grad_func, sampler_params, num_sampler_optimization_steps)
    # for i in xrange(num_sampler_optimization_steps):
    #     ml, dml = ml_and_grad(sampler_params)
    #     print "log marginal likelihood:", ml
    #     sampler_params = sampler_params + sampler_learn_rate * dml

    t1 = time.time()
    print "total runtime", t1-t0


