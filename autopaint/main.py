# Main demo script
import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
import numpy.linalg
import matplotlib.image
import pickle

from util import *

num_classes = 10

def build_image_loglik(all_mean, all_cov):
    # Define log-likelihood of natural-image prior.
    (sign, logdet) = numpy.linalg.slogdet(all_cov)
    print "logdet", logdet
    pinv = np.linalg.pinv(all_cov)
    const =  -0.5 * len(all_mean) * np.log(2*np.pi) - 0.5 * logdet
    def image_prior_log_likelihood(image):
        minus_mean = image - all_mean
        return const - 0.5 * np.dot(np.dot(minus_mean.T, pinv), minus_mean)
    return image_prior_log_likelihood


def sanity_checking_plots():
    sample_from_gaussian_model(train_images,'all')

    # Class-conditional models
    for i in range(num_classes):
        cur_class_rows = train_labels[:, i] == 1
        cur_class_images = train_images[cur_class_rows, :]
        print 'trainI shape:', cur_class_images.shape
        sample_from_gaussian_model(cur_class_images, '../figures/sample ' + str(i))


def model_mnist():
    # Load and process MNIST data
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    trained_weights = train_nn(train_images, train_labels, test_images, test_labels)
    all_mean, all_cov = mean_and_cov(train_images)
    mnist_models = trained_weights, all_mean, all_cov
    with open('mnist_models.pkl', 'w') as f:
        pickle.dump(mnist_models, f, 1)

if __name__ == '__main__':

    #model_mnist()  # Comment this out after running once.

    with open('mnist_models.pkl') as f:
        trained_weights, all_mean, all_cov = pickle.load(f)

    # Build natural image model.
    image_prior_log_likelihood = build_image_loglik(all_mean, all_cov)
    image_prior_nll = lambda i: -image_prior_log_likelihood(i)
    image_prior_with_grad = value_and_grad(image_prior_nll)

    # Build likelihood model.
    N_weights, predict_fun, loss_fun, frac_err = make_nn_funs(layer_sizes, L2_reg)
    classifier_loglik = lambda image, c: predict_fun(trained_weights, np.atleast_2d(image))[:, c]

    # Combine prior and likelihood.
    model_nll = lambda image, c: -image_prior_log_likelihood(image) - classifier_loglik(image, c)
    model_nll_with_grad = value_and_grad(model_nll)


    # Optimize a random image to maximize this likelihood.
    cur_class = 9
    start_image = np.zeros((28*28))
    quick_grad_check(image_prior_log_likelihood, start_image)

    def callback(image):
        #print "Cur loglik: ", image_prior_nll(image), "mean loglik:", image_prior_nll(all_mean)
        matplotlib.image.imsave("../figures/optimizing", image.reshape((28,28)))

    # Optimize using conjugate gradients.
    result = minimize(model_nll_with_grad, callback=callback, x0=start_image, args=(cur_class),
                      jac=True, method='BFGS')
    final_image = result.x
    matplotlib.image.imsave("../figures/optimal", final_image.reshape((28,28)))
    print "Finished!"