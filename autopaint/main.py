# Main demo script
from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.util import quick_grad_check

import matplotlib.pyplot as plt
import matplotlib.image

def make_nn_funs(layer_sizes, L2_reg):
    shapes = zip(layer_sizes[:-1], layer_sizes[1:])
    N = sum((m+1)*n for m, n in shapes)

    def unpack_layers(W_vect):
        for m, n in shapes:
            yield W_vect[:m*n].reshape((m,n)), W_vect[m*n:m*n+n]
            W_vect = W_vect[(m+1)*n:]

    def predictions(W_vect, inputs):
        for W, b in unpack_layers(W_vect):
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs - logsumexp(outputs, axis=1, keepdims=True)

    def loss(W_vect, X, T):
        log_prior = -L2_reg * np.dot(W_vect, W_vect)
        log_lik = np.sum(predictions(W_vect, X) * T)
        return - log_prior - log_lik

    def frac_err(W_vect, X, T):
        return np.mean(np.argmax(T, axis=1) != np.argmax(pred_fun(W_vect, X), axis=1))

    return N, predictions, loss, frac_err


def load_mnist():
    print("Loading training data...")
    import imp, urllib
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, K: np.array(x[:,None] == np.arange(K)[None, :], dtype=int)
    source, _ = urllib.urlretrieve(
        'https://raw.githubusercontent.com/HIPS/Kayak/master/examples/data.py')
    data = imp.load_source('data', source).mnist()
    train_images, train_labels, test_images, test_labels = data
    train_images = partial_flatten(train_images) / 255.0
    test_images  = partial_flatten(test_images)  / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels


def make_batches(N_data, batch_size):
    return [slice(i, min(i+batch_size, N_data))
            for i in range(0, N_data, batch_size)]

def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28,28),
                cmap=matplotlib.cm.binary, vmin=None):

    """iamges should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.ceil(float(N_images) / ims_per_row)
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                            (digit_dimensions[0] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i / ims_per_row  # Integer division.
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0])*row_ix
        col_start = padding + (padding + digit_dimensions[0])*col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
                      col_start: col_start + digit_dimensions[0]] \
            = cur_image
    cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    return cax


if __name__ == '__main__':
    # Network parameters
    layer_sizes = [784, 200, 100, 10]
    L2_reg = 1.0

    # Training parameters
    param_scale = 0.1
    learning_rate = 1e-3
    momentum = 0.9
    batch_size = 256
    num_epochs = 50

    # Load and process MNIST data (borrowing from Kayak)
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()

    # Make "model of natural images"
    empirical_mean = np.mean(train_images, 0)
    centered = train_images - empirical_mean
    empirical_cov = np.dot(centered.T, centered)

    # Plot the mean
    matplotlib.image.imsave('mean.png', empirical_mean.reshape((28,28)))

    # Draw a sample
    sample = np.random.multivariate_normal(empirical_mean, empirical_cov, 10).reshape((10,28*28))

    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Samples")
    plot_images(sample, ax)
    fig.set_size_inches((8,12))
    plt.savefig('sample2.png', pad_inches=0.05, bbox_inches='tight')


def nonsesnes():
    # Make neural net functions
    N_weights, pred_fun, loss_fun, frac_err = make_nn_funs(layer_sizes, L2_reg)
    loss_grad = grad(loss_fun)

    # Initialize weights
    rs = npr.RandomState()
    W = rs.randn(N_weights) * param_scale

    # Check the gradients numerically, just to be safe
    quick_grad_check(loss_fun, W, (train_images, train_labels))

    print("    Epoch      |    Train err  |   Test err  ")

    def print_perf(epoch, W):
        test_perf  = frac_err(W, test_images, test_labels)
        train_perf = frac_err(W, train_images, train_labels)
        print("{0:15}|{1:15}|{2:15}".format(epoch, train_perf, test_perf))

    # Train with sgd
    batch_idxs = make_batches(train_images.shape[0], batch_size)
    cur_dir = np.zeros(N_weights)

    for epoch in range(num_epochs):
        print_perf(epoch, W)
        for idxs in batch_idxs:
            grad_W = loss_grad(W, train_images[idxs], train_labels[idxs])
            cur_dir = momentum * cur_dir + (1.0 - momentum) * grad_W
            W -= learning_rate * cur_dir