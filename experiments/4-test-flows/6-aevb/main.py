# Main demo script
import autograd.numpy as np
from autograd import value_and_grad
import numpy.linalg
import time

from autopaint.flows import build_flow_sampler,build_batch_flow_sampler
from autopaint.plotting import plot_density
from autopaint.util import build_logprob_mvn
from autopaint.neuralnet import make_batches
from autopaint.aevb import build_encoder,build_gaussian_decoder
from autopaint.optimizers import adam_mini_batch, sga_mini_batch

def logprob_two_moons(z):
    z1 = z[:, 0]
    z2 = z[:, 1]
    return (- 0.5 * ((np.sqrt(z1**2 + z2**2) - 2 ) / 0.4)**2\
            + np.logaddexp(-0.5 * ((z1 - 2) / 0.6)**2, -0.5 * ((z1 + 2) / 0.6)**2))

def logprob_wiggle(z):
    z1 = z[:, 0]
    z2 = z[:, 1]
    return -0.5 * (z2 - np.sin(2.0 * np.pi * z1 / 4.0) / 0.4 )**2 - 0.2 * (z1**2 + z2**2)

cov = np.array([[1.0, 0.9], [0.9, 1.0]])
pinv = np.linalg.pinv(cov)
(sign, logdet) = numpy.linalg.slogdet(cov)
const =  -0.5 * 2 * np.log(2*np.pi) - 0.5 * logdet
def logprob_mvn(z):
    return const - 0.5 * np.dot(np.dot(z.T, pinv), z)


if __name__ == '__main__':

    t0 = time.time()
    rs = np.random.npr.RandomState(0)

    train_images = np.random.multivariate_normal(np.zeros(2),np.array([[1,.9],[.9,1]]),10000)

    #Create aevb function
    # Training parameters
    param_scale = 0.1
    samples_per_image = 10
    latent_dimensions = 10
    hidden_units = 50
    D = train_images.shape[1]

    enc_layers = [D, hidden_units, 2*latent_dimensions]
    dec_layers = [latent_dimensions, hidden_units, D*2]

    N_weights_enc,encoder = build_encoder(enc_layers)
    N_weights_dec,decoder, decoder_log_like = build_gaussian_decoder(dec_layers)

    #Optimize aevb
    batch_size = 3
    num_epochs = 1
    num_steps = 32
    num_sampler_optimization_steps = 400
    sampler_learn_rate = 0.01

    D = 2
    init_enc_w = rs.randn(N_weights_enc) * param_scale
    init_dec_w = rs.randn(N_weights_dec) * param_scale
    init_mean = np.zeros(latent_dimensions)
    init_log_stddevs = np.log(0.1*np.ones(latent_dimensions))
    init_output_weights = 0.1*rs.randn(num_steps, latent_dimensions)
    init_transform_weights = 0.1*rs.randn(num_steps, latent_dimensions)
    init_biases = 0.1*rs.randn(num_steps)

    #logprob_mvn = build_logprob_mvn(mean=np.array([0.2,0.4]), cov=np.array([[1.0,0.9], [0.9,1.0]]))
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


    def get_batch_marginal_likelihood_estimate(sampler_params,idxs):
        #Create function to compute lower_bound
        def get_batch_lower_bound(sampler_params):
            #Create an initial encoding of sample images:
            enc_w = parser.get(sampler_params,'enc_w')
            (mus,log_sigs) = encoder(enc_w,train_images[idxs])
            #Take mean of encodings and sigs and use this to generate samples, should return only entropy
            samples, entropy_estimates = flow_sample(sampler_params, mus, np.exp(log_sigs),rs, samples_per_image)
            #From samples decode them and compute likelihood
            dec_w = parser.get(sampler_params,'dec_w')
            train_images_repeat = np.repeat(train_images[idxs],samples_per_image,axis=0)
            loglike = decoder_log_like(dec_w,samples,train_images_repeat)
            print "Mean loglik:", loglike.value,\
            "Mean entropy:", np.mean(entropy_estimates.value)
            #Create some samples from x:
            (xs_mu,xs_log_sigs) = decoder(dec_w, samples)
            # plot_density(xs_mu.value, "approximating_dist.png")
            return np.mean(entropy_estimates)
            print np.mean(entropy_estimates)+loglike
            return np.mean(entropy_estimates)+loglike

        #Take gradient
        lb_val_grad = value_and_grad(get_batch_lower_bound)
        return lb_val_grad(sampler_params)

    def print_ml(ml,weights,grad):
        print "log marginal likelihood:", ml


    final_params, final_value = adam_mini_batch(get_batch_marginal_likelihood_estimate,sampler_params,batch_idxs,num_epochs,callback=print_ml)


    t1 = time.time()
    print "total runtime", t1-t0


