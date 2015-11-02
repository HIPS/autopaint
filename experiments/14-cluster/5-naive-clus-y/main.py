import pickle
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as kmeans
from sklearn.decomposition import PCA
import autograd.numpy.random as npr
import autograd.numpy as np



from autograd import grad

from autopaint.plotting import plot_images
from autopaint.optimizers import adam
from autopaint.neuralnet import make_batches
from autopaint.util import load_mnist
from autopaint.aevb import lower_bound
from autopaint.util import WeightsParser, load_and_pickle_binary_mnist
from autopaint.neuralnet import make_binary_nn,make_gaussian_nn
param_scale = 0.1
samples_per_image = 1
latent_dimensions = 10
hidden_units = 300

def get_pretrained_nn_weights():
   with open('parameters40l300hfor3000.pkl') as f:
        parameters = pickle.load(f)
   params,N_weights_enc,samples_per_image,latent_dimensions,rs = parameters
   return params

def plot_centers(train_images):
    im_clus = kmeans(10)
    im_clus.fit(train_images)
    centers = im_clus.cluster_centers_[:,0:784]
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(centers, ax, ims_per_row=10)
    plt.savefig('centroid.png')


if __name__ == '__main__':
    # load_and_pickle_binary_mnist()
    with open('../../../autopaint/mnist_binary_data.pkl') as f:
        N_data, train_images, train_labels, test_images, test_labels = pickle.load(f)

    train = np.concatenate((train_images,train_labels),axis = 1)
    plot_centers(train)

