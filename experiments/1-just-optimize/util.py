# Main demo script
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.util import quick_grad_check

import matplotlib.pyplot as plt
import matplotlib.image


def entropy_of_a_gaussian(stddevs):
    # TODO: double check this formula.
    D = len(stddevs)
    0.05 * D * (1 + np.log(2*np.pi)) + np.sum(np.log(stddevs))

def entropy_of_a_spherical_gaussian(stddev, D):
    # TODO: double check this formula.
    0.05 * D * (1 + np.log(2*np.pi)) + D * np.log(stddev)

def sgd_entropic(gradfun, x, N_iter, learn_rate, rs, callback, approx=True):
    D = len(x)

    delta_entropy = 0.0
    for t in xrange(N_iter):
        g = gradfun(x, t)
        hvp = grad(lambda x, vect : np.dot(gradfun(x, t), vect)) # Hessian vector product
        jvp = lambda vect : vect - learn_rate * hvp(x, vect) # Jacobian vector product
        if approx:
            delta_entropy += approx_log_det(jvp, D, rs)
        else:
            delta_entropy += exact_log_det(jvp, D, rs)
        if callback: callback(x=x, t=t, entropy=delta_entropy)
        x -= learn_rate * g

    return x, delta_entropy


def sgd_entropic(gradfun, x_scale, N_iter, learn_rate, rs, callback, approx=True):
    D = len(x_scale)
    x = rs.randn(D) * x_scale
    entropy = 0.5 * D * (1 + np.log(2*np.pi)) + np.sum(np.log(x_scale))
    for t in xrange(N_iter):
        g = gradfun(x, t)
        hvp = grad(lambda x, vect : np.dot(gradfun(x, t), vect)) # Hessian vector product
        jvp = lambda vect : vect - learn_rate * hvp(x, vect) # Jacobian vector product
        if approx:
            entropy += approx_log_det(jvp, D, rs)
        else:
            entropy += exact_log_det(jvp, D, rs)
        if callback: callback(x=x, t=t, entropy=entropy)
        x -= learn_rate * g

    return x, entropy

def approx_log_det(mvp, D, rs):
    # This should be an unbiased estimator of a lower bound on the log determinant
    # provided the eigenvalues are all greater than 0.317 (solution of
    # log(x) = (x - 1) - (x - 1)**2 = -2 + 3 * x - x**2
    R0 = rs.randn(D) # TODO: Consider normalizing R
    R1 = mvp(R0)
    R2 = mvp(R1)
    return np.dot(R0, -2 * R0 + 3 * R1 - R2)

def exact_log_det(mvp, D, rs=None):
    mat = np.zeros((D, D))
    eye = np.eye(D)
    for i in range(D):
        mat[:, i] = mvp(eye[:, i])
    return np.log(np.linalg.det(mat))


def make_nn_funs(layer_sizes, L2_reg):
    shapes = zip(layer_sizes[:-1], layer_sizes[1:])
    N = sum((m+1)*n for m, n in shapes)

    def unpack_layers(W_vect):
        for m, n in shapes:
            yield W_vect[:m*n].reshape((m,n)), W_vect[m*n:m*n+n]
            W_vect = W_vect[(m+1)*n:]

    def predict_fun(W_vect, inputs):
        """Returns normalized log-prob of all classes."""
        for W, b in unpack_layers(W_vect):
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs - logsumexp(outputs, axis=1, keepdims=True)

    def loss(W_vect, X, T):
        log_prior = -L2_reg * np.dot(W_vect, W_vect)
        log_lik = np.sum(predict_fun(W_vect, X) * T)
        return - log_prior - log_lik

    def frac_err(W_vect, X, T):
        return np.mean(np.argmax(T, axis=1) != np.argmax(predict_fun(W_vect, X), axis=1))

    return N, predict_fun, loss, frac_err


def load_mnist():
    print "Loading training data..."
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

def mean_and_cov(images):
    # Make "model of natural images"
    empirical_mean = np.mean(images, 0)
    centered = images - empirical_mean
    empirical_cov = np.dot(centered.T, centered) + 0.001 * np.eye(len(empirical_mean))
    return empirical_mean, empirical_cov

def plot_samples(mean, cov, file_prefix):
    # Plot the mean
    meanName = file_prefix + 'mean.png'
    matplotlib.image.imsave(meanName, mean.reshape((28,28)))

    # Draw a sample
    sample = np.random.multivariate_normal(mean, cov, 10).reshape((10,28*28))

    fig = plt.figure(0)
    fig.clf()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Samples")
    plot_images(sample, ax)
    fig.set_size_inches((8,12))
    sampleName = file_prefix + 'sample.png'
    plt.savefig(sampleName, pad_inches=0.05, bbox_inches='tight')

def sample_from_gaussian_model(images, prefix):
    mean, cov = mean_and_cov(images)
    plot_samples(mean, cov, prefix)

def sgd(grad, x, callback=None, num_iters=200, step_size=0.1, mass=0.9):
    """Stochastic gradient descent with momentum.
    grad() must have signature grad(x, i), where i is the iteration number."""
    velocity = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x)
        if callback: callback(x, i, g)
        velocity = mass * velocity - (1.0 - mass) * g
        x += step_size * velocity
    return x

# Network parameters
layer_sizes = [784, 200, 100, 10]
L2_reg = 1.0

# Training parameters
param_scale = 0.1
learning_rate = 1e-3
momentum = 0.9
batch_size = 256
num_epochs = 50

def train_nn(train_images, train_labels, test_images, test_labels):

    # Make neural net functions
    N_weights, predict_fun, loss_fun, frac_err = make_nn_funs(layer_sizes, L2_reg)
    loss_grad = grad(loss_fun)

    # Initialize weights
    rs = npr.RandomState()
    weights = rs.randn(N_weights) * param_scale

    # Check the gradients numerically, just to be safe
    quick_grad_check(loss_fun, weights, (train_images, train_labels))

    print "    Epoch      |    Train err  |   Test err  "

    def print_perf(epoch, weights):
        test_perf  = frac_err(weights, test_images, test_labels)
        train_perf = frac_err(weights, train_images, train_labels)
        print "{0:15}|{1:15}|{2:15}".format(epoch, train_perf, test_perf)

    # Train with sgd
    batch_idxs = make_batches(train_images.shape[0], batch_size)
    cur_dir = np.zeros(N_weights)

    for epoch in range(num_epochs):
        print_perf(epoch, weights)
        for idxs in batch_idxs:
            grad_W = loss_grad(weights, train_images[idxs], train_labels[idxs])
            cur_dir = momentum * cur_dir + (1.0 - momentum) * grad_W
            weights -= learning_rate * cur_dir

    return weights


def plot_2d_func(energy_func, filename, xlims=[-4.0, 4.0], ylims = [-4.0, 4.0]):
    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    x = np.linspace(*xlims, num=100)
    y = np.linspace(*ylims, num=100)
    X, Y = np.meshgrid(x, y)
    zs = np.array([energy_func(np.concatenate(([x],[y]))) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    Z = np.exp(-Z)
    matplotlib.image.imsave(filename, Z)

def plot_density(samples, filename, xlims=[-4.0, 4.0], ylims = [-4.0, 4.0]):
    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    x = np.linspace(*xlims, num=100)
    y = np.linspace(*ylims, num=100)
    plt.scatter(samples[:,0], samples[:,1])
    plt.savefig(filename)
