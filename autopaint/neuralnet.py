import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad
from autograd.util import quick_grad_check
from autopaint.util import sigmoid

def make_nn_funs(layer_sizes):
    shapes = zip(layer_sizes[:-1], layer_sizes[1:])
    num_weights = sum((m+1)*n for m, n in shapes)

    def unpack_layers(weights):
        for m, n in shapes:
            yield weights[:m*n].reshape((m,n)), weights[m*n:m*n+n]
            weights = weights[(m+1)*n:]

    def compute_hiddens(weights, inputs):
        """Returns normalized log-prob of all classes."""
        for layer, (W, b) in enumerate(unpack_layers(weights)):
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs

    return num_weights, compute_hiddens


def make_classification_nn(layer_sizes):
    """Outputs class label log-probabilities."""
    num_weights, compute_hiddens = make_nn_funs(layer_sizes)

    def make_predictions(weights, inputs):
        """Normalize log-probabilities."""
        hiddens = compute_hiddens(weights, inputs)
        return hiddens - logsumexp(hiddens, axis=1, keepdims=True)

    def likelihood(weights, inputs, targets):
        return np.sum(make_predictions(weights, inputs) * targets, axis=1)

    return num_weights, make_predictions, likelihood


def make_binarized_nn_funs(layer_sizes):
    """Outputs are in [0,1]^D and the labels are {0,1}^D"""
    num_weights, compute_hiddens = make_nn_funs(layer_sizes)

    def make_predictions(weights, inputs):
        return sigmoid(compute_hiddens(weights, inputs))

    def likelihood(weights, inputs, targets):
        pred_probs = make_predictions(weights, inputs)
        label_probabilities = np.log(pred_probs)       * targets \
                            + np.log((1 - pred_probs)) * (1 - targets)
        return np.sum(label_probabilities, axis=1)   # Sum across pixels.

    return num_weights, make_predictions, likelihood


def make_gaussian_nn_funs(layer_sizes):
    """Outputs a Guassian."""
    num_weights, compute_hiddens = make_nn_funs(layer_sizes)
    D = layer_sizes[-1] / 2

    def make_predictions(weights, inputs):
        """Returns the mean and the log of the diagonal of the covariance matrix."""
        hiddens = compute_hiddens(weights, inputs)
        mu = hiddens[:, 0:D]
        log_sig = hiddens[:, D:2*D]
        return mu,log_sig

    def likelihood(weights, inputs, targets):
        means, log_sigs = make_predictions(weights, inputs)
        normalized_targets = (targets - means) / log_sigs
        const = -0.5 * D * np.log(2*np.pi) - np.sum(log_sigs, axis=1)
        return const - 0.5 * np.einsum('ij,ij->i', normalized_targets, normalized_targets)

    return num_weights, make_predictions, likelihood


one_hot = lambda x, K: np.array(x[:,None] == np.arange(K)[None, :], dtype=int)

def load_mnist():
    print "Loading training data..."
    import imp, urllib
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))

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


def train_nn(train_images, train_labels, test_images, test_labels):
    # Network parameters   TODO: move these into experiment scripts.
    layer_sizes = [784, 200, 100, 10]
    L2_reg = 1.0

    # Training parameters
    param_scale = 0.1
    learning_rate = 1e-3
    momentum = 0.9
    batch_size = 256
    num_epochs = 50

    # Make neural net functions
    N_weights, predict_fun, loss_fun, frac_err, likelihood = make_nn_funs(layer_sizes, L2_reg)
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

    return weights, predict_fun, likelihood


