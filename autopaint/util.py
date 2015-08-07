import autograd.numpy as np
import scipy.linalg

class WeightsParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.num_weights = 0

    def add_shape(self, name, shape):
        start = self.num_weights
        self.num_weights += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.num_weights), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)

    def put(self, vect, name, val):
        idxs, shape = self.idxs_and_shapes[name]
        vect[idxs].reshape(shape)[:] = val

    def __len__(self):
        return self.num_weights


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


def mean_and_cov(images):
    # Make "model of natural images"
    empirical_mean = np.mean(images, 0)
    centered = images - empirical_mean
    empirical_cov = np.dot(centered.T, centered) + 0.001 * np.eye(len(empirical_mean))
    return empirical_mean, empirical_cov


def sample_from_gaussian_model(images, prefix):
    mean, cov = mean_and_cov(images)
    plot_samples(mean, cov, prefix)

def fast_array_from_list(xs):
    return np.concatenate([np.expand_dims(x, axis=0) for x in xs], axis=0)


# Some example test likelihood functions

def logprob_two_moons(z):
    z1 = z[:, 0]
    z2 = z[:, 1]
    return (- 0.5 * ((np.sqrt(z1**2 + z2**2) - 2 ) / 0.4)**2\
            + np.logaddexp(-0.5 * ((z1 - 2) / 0.6)**2, -0.5 * ((z1 + 2) / 0.6)**2))

def logprob_wiggle(z):
    z1 = z[:, 0]
    z2 = z[:, 1]
    return -0.5 * (z2 - np.sin(2.0 * np.pi * z1 / 4.0) / 0.4 )**2 - 0.2 * (z1**2 + z2**2)

def log_normalizing_constant_of_a_guassian(cov):
    D = cov.shape[0]
    (sign, logdet) = np.linalg.slogdet(cov)
    return -0.5 * D * np.log(2*np.pi) - 0.5 * logdet

def build_logprob_mvn(mean, cov):
    pinv = np.linalg.pinv(cov)
    const = log_normalizing_constant_of_a_guassian(cov)
    def logprob_mvn(z):
        """z is NxD."""
        z_minus_mean = z - mean
        return const - 0.5 * np.einsum('ij,jk,ik->i', z_minus_mean, pinv, z_minus_mean)
    return logprob_mvn

def build_logprob_standard_normal(D):
    const = log_normalizing_constant_of_a_guassian(np.eye(D))
    def logprob(z):
        """z is NxD."""
        return const - 0.5 * np.einsum('ij,ij->i', z, z)
    return logprob

def build_unwhitener(mean, cov):
    """Builds a function that takes in a draw from a standard normal, and
       turns it into a draw from a MVN with mean, cov."""
    #chol = np.linalg.cholesky(cov)
    sq = np.real(scipy.linalg.sqrtm(cov))
    def unwhitener(z):
        return np.dot(z, sq) + mean
    return unwhitener

def entropy_of_a_gaussian(cov):
    D = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    return 0.5 * D * (1.0 + np.log(2*np.pi)) + 0.5 * logdet

def entropy_of_a_diagonal_gaussian(stddevs):
    D = len(stddevs)
    return 0.5 * D * (1.0 + np.log(2*np.pi)) + np.sum(np.log(stddevs))

def entropy_of_a_spherical_gaussian(stddev, D):
    return 0.5 * D * (1.0 + np.log(2*np.pi)) + D * np.log(stddev)

def low_variance_gradient_estimator(grad_entropies, grad_likelihoods):
    """Produces a lower-variance estimate of the gradient w.r.t. the variational lower bound
        by exploiting the fact that E_q(gradient of log(q)) = 0 for any q.
        From eqn (8) of http://arxiv.org/pdf/1401.1022v3.pdf"""
    empirical_entropy_grad_mean = np.mean(grad_entropies)
    return np.mean((grad_entropies - empirical_entropy_grad_mean)*(grad_entropies + grad_likelihoods))