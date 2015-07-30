import autograd.numpy as np
from autograd import grad


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

