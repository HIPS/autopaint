import autograd.numpy as np

import matplotlib.pyplot as plt
import matplotlib.image


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
    #fig.set_size_inches((8,12))
    sampleName = file_prefix + 'sample.png'
    plt.savefig(sampleName, pad_inches=0.05, bbox_inches='tight')


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


def plot_density(samples, filename):
    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    plt.scatter(samples[:,0], samples[:,1])
    plt.savefig(filename)


def plot_line(x,y, filename):
    fig = plt.figure(0); fig.clf()
    ax = fig.add_subplot(111)
    plt.plot(x, y)
    plt.savefig(filename)
