import autograd.numpy as np
from autopaint.util import entropy_of_a_diagonal_gaussian
from plotting import plot_density

def lower_bound(weights,encode,decode_log_like,log_prior,N_weights_enc,train_images,samples_per_image,latent_dimensions,rs):
    enc_w = weights[0:N_weights_enc]
    dec_w = weights[N_weights_enc:len(weights)]
    mean_log_joint,mean_ent = compute_log_prob_and_ent(enc_w,dec_w,encode,decode_log_like,log_prior,train_images,samples_per_image,latent_dimensions,rs)
    print "joint ll average",mean_log_joint
    print "ent average", mean_ent
    return mean_log_joint + mean_ent

def compute_log_prob_and_ent(enc_w,dec_w,encode,decode_log_like,log_prior,train_images,samples_per_image,latent_dimensions,rs):
    (mus,log_sigs) = encode(enc_w,train_images)
    sigs = np.exp(log_sigs)
    sigs = np.ones(sigs.shape)
    ents = entropy_of_a_diagonal_gaussian(sigs)
    mean_ent = np.mean(ents)
    noise = rs.randn(samples_per_image,train_images.shape[0],latent_dimensions)
    Z_samples = mus + sigs*noise
    Z_samples = np.reshape(Z_samples,(train_images.shape[0]*samples_per_image,latent_dimensions),order = 'F')
    train_images_repeat = np.repeat(train_images,samples_per_image,axis=0)
    print 'prior avg', np.mean(log_prior(Z_samples))
    try:
        plot_samples = Z_samples.value
    except:
        plot_samples = Z_samples
    plot_density(plot_samples,'z_samples.png')
    mean_log_prob = np.mean(decode_log_like(dec_w,Z_samples,train_images_repeat) +log_prior(Z_samples))
    return mean_log_prob, mean_ent



