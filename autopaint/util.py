import pickle
import autograd.numpy as np
from scipy.linalg import sqrtm
# from autograd.scipy.special import gammaln

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

def log_inv_rosenbrock(z):
    #z is (N,D)
    #Returns (N,) of log( 1/(rosenbrock(z)+relaxation))
    relaxation = 1e-6
    (N,D) = z.shape
    s = np.zeros(N)
    for i in xrange(D-1):
        s = s + (1-z[:,i])**2+100*(z[:,i+1]-z[:,i]**2)**2
    result = np.log(1/(s+relaxation))
    return result

def log_tapered_inv_rosenbrock(z):
    #z is (N,D)
    #Returns s (N,) tapers inv_ros by sq exp to prevent entropy from exploding as D grows
    relaxation = 1e-6
    (N,D) = z.shape
    scale = 1e-4
    s = np.zeros(N)
    for i in xrange(D-1):
        s = s + (1-z[:,i])**2+100*(z[:,i+1]-z[:,i]**2)**2
    result = np.log(1/(s+relaxation)*np.exp(-scale*np.sum(z**2)))
    return result

def log_normalizing_constant_of_a_guassian(cov):
    D = cov.shape[0]
    (sign, logdet) = np.linalg.slogdet(cov)
    return -0.5 * D * np.log(2*np.pi) - 0.5 * logdet

def build_logprob_mvn(mean, cov,pseudo_inv = True):
    if pseudo_inv == True:
        pinv = np.linalg.pinv(cov)
    else:
        pinv = np.linalg.inv(cov)
    const = log_normalizing_constant_of_a_guassian(cov)
    def logprob(z):
        """z is NxD."""
        z_minus_mean = z - mean
        if len(z.shape) == 1 or z.shape[0] == 1:
            return const - 0.5*np.dot(np.dot(z_minus_mean,pinv),z_minus_mean.T)
        else:
            return const - 0.5 * np.einsum('ij,jk,ik->i', z_minus_mean, pinv, z_minus_mean)

    return logprob

def log_normalizing_constant_of_a_mvt(cov, dof):
    D = cov.shape[0]
    (sign, logdet) = np.linalg.slogdet(cov)
    return gammaln((dof + D) / 2.0) - gammaln(dof / 2.0) - 0.5 * D * np.log(dof) \
           - 0.5 * D * np.log(np.pi) - 0.5 * logdet

def build_logprob_mvt(mean, cov, dof):
    """Logprob of multivariate student's t"""
    pinv = np.linalg.pinv(cov)
    const = log_normalizing_constant_of_a_mvt(cov, dof)
    D = len(mean)
    def logprob(z):
        """z is NxD."""
        z_minus_mean = z - mean
        mahalanobois_dist = np.einsum('ij,jk,ik->i', z_minus_mean, pinv, z_minus_mean)
        return const - 0.5 * (dof - D) * np.logaddexp(1.0, (1.0/dof)*mahalanobois_dist)
    return logprob

def build_logprob_standard_normal(D):
    const = log_normalizing_constant_of_a_guassian(np.eye(D))
    def logprob(z):
        """z is NxD."""
        return const - 0.5 * np.einsum('ij,ij->i', z, z)
    return logprob

def sample_from_normal_bimodal(mean1,mean2,num_samples,rs):
    D = len(mean1)
    samples = np.zeros((num_samples,D))
    for i in xrange(num_samples):
        z = rs.randn()
        noise = rs.randn(D)
        if z < .5:
            samples[i,:] = mean1+noise
        else:
            samples[i,:] = mean2+noise
    return samples


def build_unwhitener(mean, cov):
    """Builds a function that takes in a draw from a standard normal, and
       turns it into a draw from a MVN with mean, cov."""
    #chol = np.linalg.cholesky(cov)
    sq = np.real(sqrtm(cov))
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

def entropy_of_diagonal_gaussians(stddevs_mat):
    #Returns entropy of several different diagonal gaussians
    if len(stddevs_mat.shape) == 1:
        stddevs_mat = np.atleast_2d(stddevs_mat)
    D = stddevs_mat.shape[1]
    return 0.5 * D * (1.0 + np.log(2*np.pi)) + np.sum(np.log(stddevs_mat), axis=1)


def entropy_of_a_spherical_gaussian(stddev, D):
    return 0.5 * D * (1.0 + np.log(2*np.pi)) + D * np.log(stddev)

def low_variance_gradient_estimator(grad_entropies, grad_likelihoods):
    """Produces a lower-variance estimate of the gradient w.r.t. the variational lower bound
        by exploiting the fact that E_q(gradient of log(q)) = 0 for any q.
        From eqn (8) of http://arxiv.org/pdf/1401.1022v3.pdf"""
    empirical_entropy_grad_mean = np.mean(grad_entropies)
    return np.mean((grad_entropies - empirical_entropy_grad_mean)*(grad_entropies + grad_likelihoods))

def sigmoid(x):
    return 0.5*(np.tanh(x) + 1.0)

def inv_sigmoid(x):
    return np.arctanh(2.0 * x - 1.0)

def approx_log_det(mvp_vec, D, N, rs):
    # This should be an unbiased estimator of a lower bound on the log determinant
    # provided the eigenvalues are all greater than 0.317 (solution of
    # log(x) = (x - 1) - (x - 1)**2 = -2 + 3 * x - x**2
    # This vectorized version computes N independent bound estimates.
    R0 = rs.randn(N, D) # TODO: Consider normalizing R.
    R1 = mvp_vec(R0)
    R2 = mvp_vec(R1)
    return np.sum(R0 * (-2 * R0 + 3 * R1 - R2), axis=1)  # Row-wise dot products.


def approx_log_det_vectorized_avg(mvp_vec, D, N, num_samples, rs):
    """Averages over several random projections."""
    # Can this be vectorized more directly, without a for loop?
    approx_logdets = [approx_log_det(mvp_vec, D, N, rs) for n in xrange(num_samples)]
    approx_logdets = fast_array_from_list(approx_logdets)
    return np.mean(approx_logdets, axis=0)


def exact_log_det(mvp_vec, D, N):
    """mvp_vec is a function that takes in a matrix of size N x D and returns another matrix of size N x D.
    This function builds N Jacobians explicitly, and returns a vector representing the logdets of these Jacobians."""
    eye = np.eye(D)
    matlist = []
    for i in xrange(D):
        cur_dir = eye[:, i]
        matlist.append(mvp_vec(np.tile(cur_dir, (N, 1))))
    mat = np.concatenate([np.expand_dims(x, axis=2) for x in matlist], axis=2)
    logdets = []
    for cur_mat in mat:  # Not vectorized, but could be if autograd supported vectorized calls to slogdet.  Does it?
        sign, logdet = np.linalg.slogdet(cur_mat)
        logdets.append(logdet)
    assert len(logdets) == N
    return fast_array_from_list(logdets)


def sum_entropy_lower_bound(entropy_a, entropy_b, D):
    """Returns lower bound of X + Y given the entropy of X and Y.
    Uses the entropy power inequality.
    https://en.wikipedia.org/wiki/Entropy_power_inequality"""
    return 0.5 * D * np.logaddexp(2.0 * entropy_a / D, 2.0 * entropy_b / D)


def neg_kl_diag_normal(mu,sig):
    #Computes of the -1 * kl divergence of a diagonal gaussians vs a normal gaussian
    #Takes in an nxd vectors of means and diagonal covariances
    D = mu.shape[1]
    combined_mat = np.log(sig**2)-mu**2-sig**2
    kl_vect = np.sum(combined_mat,axis = 1)
    kl_vect = kl_vect + D*np.ones(mu.shape[0])
    kl_vect = .5*kl_vect
    return kl_vect

def neg_kl_diag_scaled_normal(mu,sig,alpha):
    #Computes of the -1 * kl divergence of a diagonal gaussians vs an alpha*normal gaussian
    #Takes in an nxd vectors of means and diagonal covariances
    D = mu.shape[1]
    combined_mat = np.log(sig**2)-(mu**2+sig**2)/alpha**2
    kl_vect = np.sum(combined_mat,axis = 1)
    kl_vect = kl_vect + D*np.ones(mu.shape[0])-D*np.log(alpha**2)*np.ones(mu.shape[0])
    kl_vect = .5*kl_vect
    return kl_vect

def binarized_loglike(pred_probs,T):
    label_probabilities =  pred_probs* T + (1 - pred_probs) * (1 - T)
    #TODO: Mean or sum?
    ll_vect = np.sum(label_probabilities,axis = 1)
    return np.mean(ll_vect)

def load_and_pickle_binary_mnist():
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()
    train_images = np.round(train_images)
    test_images = np.round(test_images)
    mnist_data = N_data, train_images, train_labels, test_images, test_labels
    with open('mnist_binary_data.pkl', 'w') as f:
        pickle.dump(mnist_data, f, 1)