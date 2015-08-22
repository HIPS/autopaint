# Main demo script
import time
import pickle
import autograd.numpy as np
from autograd import value_and_grad

from autopaint.flows import build_flow_sampler,build_batch_flow_sampler
from autopaint.plotting import plot_density
from autopaint.neuralnet import make_batches
from autopaint.aevb import build_encoder,build_binarized_decoder
from autopaint.optimizers import adam_mini_batch, sga_mini_batch
from autopaint.util import sample_from_normal_bimodal,load_and_pickle_binary_mnist

#AEVB params
param_scale = 0.1
samples_per_image = 1
latent_dimensions = 2
hidden_units = 500

def get_batch_lower_bound(cur_sampler_params, idxs):
    #Create an initial encoding of sample images:
    enc_w = parser.get(cur_sampler_params,'enc_w')
    (mus,log_sigs) = encoder(enc_w,train_images[idxs])
    #Take mean of encodings and sigs and use this to generate samples, should return only entropy
    samples, entropy_estimates = flow_sample(cur_sampler_params, mus, np.exp(log_sigs),rs, samples_per_image)
    #From samples decode them and compute likelihood
    dec_w = parser.get(cur_sampler_params,'dec_w')
    train_images_repeat = np.repeat(train_images[idxs],samples_per_image,axis=0)
    loglike = decoder_log_like(dec_w,samples,train_images_repeat)
    print "Mean loglik:", loglike.value,\
    "Mean entropy:", np.mean(entropy_estimates.value)
    return np.mean(entropy_estimates)+loglike

def print_ml(ml,weights,grad):
    print "log marginal likelihood:", ml

#
# def generate_samples(sampler_params,num_x_samples):
#     x_samples = np.zeros(num_x_samples,D)
#     for i in xrange(x_samples):
#         #Create an initial encoding of sample images:
#         enc_w = parser.get(cur_sampler_params,'enc_w')
#         (mus,log_sigs) = encoder(enc_w,train_images[idxs])
#         #Take mean of encodings and sigs and use this to generate samples, should return only entropy
#         samples, entropy_estimates = flow_sample(cur_sampler_params, mus, np.exp(log_sigs),rs, samples_per_image)
#         #From samples decode them and compute likelihood
#         dec_w = parser.get(cur_sampler_params,'dec_w')
#         train_images_repeat = np.repeat(train_images[idxs],samples_per_image,axis=0)
#         loglike = decoder_log_like(dec_w,samples,train_images_repeat)
#         print "Mean loglik:", loglike.value,\
#         "Mean entropy:", np.mean(entropy_estimates.value)
#         #Create some samples from x:
#         (xs_mu,xs_log_sigs) = decoder(dec_w, samples)
#         x_noise = rs.randn(xs_mu.shape[0],xs_mu.shape[1])
#         x_samples[i,:] = xs_mu+xs_log_sigs*x_noise
#     plot_density(x_samples.value, "approximating_dist.png")

    #
    # #Create some samples from x:
    # (xs_mu,xs_log_sigs) = decoder(dec_w, samples)
    # x_noise = rs.randn(xs_mu.shape[0],xs_mu.shape[1])
    # xs_samples = xs_mu+xs_log_sigs*x_noise
    # plot_density(xs_samples.value, "approximating_dist.png")
    # #print np.mean(entropy_estimates)+loglike

if __name__ == '__main__':

    t0 = time.time()
    rs = np.random.npr.RandomState(0)
    # load_and_pickle_binary_mnist()
    with open('../../../autopaint/mnist_binary_data.pkl') as f:
        N_data, train_images, train_labels, test_images, test_labels = pickle.load(f)

    #Create aevb function
    D = train_images.shape[1]

    enc_layers = [D, hidden_units, 2*latent_dimensions]
    dec_layers = [latent_dimensions, hidden_units, D]

    N_weights_enc,encoder = build_encoder(enc_layers)
    N_weights_dec,decoder, decoder_log_like = build_binarized_decoder(dec_layers)

    #Optimize aevb
    batch_size = 100
    num_epochs = 1
    num_steps = 32
    num_sampler_optimization_steps = 400
    sampler_learn_rate = 0.01

    init_enc_w = rs.randn(N_weights_enc) * param_scale
    init_dec_w = rs.randn(N_weights_dec) * param_scale
    init_output_weights = 0.1*rs.randn(num_steps, latent_dimensions)
    init_transform_weights = 0.1*rs.randn(num_steps, latent_dimensions)
    init_biases = 0.1*rs.randn(num_steps)

    flow_sample, parser = build_batch_flow_sampler(latent_dimensions, num_steps,batch_size)
    parser.add_shape('enc_w',len(init_enc_w))
    parser.add_shape('dec_w',len(init_dec_w))

    sampler_params = np.zeros(len(parser))
    parser.put(sampler_params, 'output weights', init_output_weights)
    parser.put(sampler_params, 'transform weights', init_transform_weights)
    parser.put(sampler_params, 'biases', init_biases)
    parser.put(sampler_params,'enc_w',init_enc_w)
    parser.put(sampler_params,'dec_w', init_dec_w)
    batch_idxs = make_batches(train_images.shape[0], batch_size)

    # get_batch_lower_bound(sampler_params,range(100))
    lb_val_grad = value_and_grad(get_batch_lower_bound)

    final_params, final_value = adam_mini_batch(lb_val_grad,sampler_params,batch_idxs,num_epochs,callback=print_ml)

    t1 = time.time()
    print "total runtime", t1-t0


