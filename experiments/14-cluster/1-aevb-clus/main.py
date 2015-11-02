import pickle
import time
import matplotlib.pyplot as plt
# from scipy.cluster.vq import kmeans2 as kmeans
from sklearn.cluster import KMeans as kmeans
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
latent_dimensions = 40
hidden_units = 300

def get_pretrained_nn_weights():
   with open('parameters40l300hfor3000.pkl') as f:
        parameters = pickle.load(f)
   params,N_weights_enc,samples_per_image,latent_dimensions,rs = parameters
   return params

def plot_latent_centers(encoder,decoder,enc_w,dec_w):
    latent_images = encoder(enc_w,train_images)[0]
    im_clus = kmeans(10)
    im_clus.fit(latent_images)
    centers = im_clus.cluster_centers_
    im_cents = decoder(dec_w,centers)
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(im_cents, ax, ims_per_row=10)
    plt.savefig('centroid.png')

def run_aevb(train_images):
    start_time = time.time()

    # Create aevb function
    # Training parameters

    D = train_images.shape[1]

    enc_layers = [D, hidden_units, hidden_units, 2*latent_dimensions]
    dec_layers = [latent_dimensions, hidden_units,hidden_units, D]

    N_weights_enc, encoder, encoder_log_like = make_gaussian_nn(enc_layers)
    N_weights_dec, decoder, decoder_log_like = make_binary_nn(dec_layers)

    # Optimize aevb
    batch_size = 500
    num_training_iters = 1600
    rs = npr.RandomState(0)

    parser = WeightsParser()
    parser.add_shape('encoding weights', (N_weights_enc,))
    parser.add_shape('decoding weights', (N_weights_dec,))
    initial_combined_weights = rs.randn(len(parser)) * param_scale

    batch_idxs = make_batches(train_images.shape[0], batch_size)

    def batch_value_and_grad(weights, iter):
        iter = iter % len(batch_idxs)
        cur_data = train_images[batch_idxs[iter]]
        return lower_bound(weights,encoder,decoder_log_like,N_weights_enc,cur_data,samples_per_image,latent_dimensions,rs)

    lb_grad = grad(batch_value_and_grad)

    big_batch_idxs = make_batches(train_images.shape[0], 1000)

    def big_batch_value_and_grad(weights, iter):
        iter = iter % len(big_batch_idxs)
        cur_data = train_images[big_batch_idxs[iter]]
        return lower_bound(weights,encoder,decoder_log_like,N_weights_enc,cur_data,samples_per_image,latent_dimensions,rs)


    def callback(params, i, grad):
        ml = big_batch_value_and_grad(params,i)
        print "log marginal likelihood:", ml


        print "----- iter ", i
        if i % 1000 == 0 and not np.isnan(lower_bound(params,encoder,decoder_log_like,N_weights_enc,test_images[0:100,:],samples_per_image,latent_dimensions,rs)):
            print 'SAVING ==== '
            save_string = 'parameters10l300hfor' +str(i) +'.pkl'

            parameters = params,N_weights_enc,samples_per_image,latent_dimensions,rs
            print 'SAVING AS: ', save_string
            print 'LATENTS DIMS', latent_dimensions
            with open(save_string, 'w') as f:
                pickle.dump(parameters, f, 1)
            #Validation loss:
            print '--- test loss:', lower_bound(params,encoder,decoder_log_like,N_weights_enc,test_images[0:100,:],samples_per_image,latent_dimensions,rs)

        #Generate samples
        num_samples = 100
        images_per_row = 10
        zs = rs.randn(num_samples,latent_dimensions)
        samples = decoder(parser.get(params, 'decoding weights'), zs)
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
        plot_images(samples, ax, ims_per_row=images_per_row)
        plt.savefig('samples.png')
        if i%100 == 0:
            enc_w = params[0:N_weights_enc]
            dec_w = params[N_weights_enc:len(params)]
            plot_latent_centers(encoder,decoder,enc_w,dec_w)

    final_params = adam(lb_grad, initial_combined_weights, num_training_iters, callback=callback)


    finish_time = time.time()
    print "total runtime", finish_time - start_time

    enc_w = final_params[0:N_weights_enc]
    dec_w = final_params[N_weights_enc:len(final_params)]
    return encoder,decoder, enc_w,dec_w




if __name__ == '__main__':
    # load_and_pickle_binary_mnist()
    with open('../../../autopaint/mnist_binary_data.pkl') as f:
        N_data, train_images, train_labels, test_images, test_labels = pickle.load(f)

    encoder,decoder,enc_w,dec_w = run_aevb(train_images)

