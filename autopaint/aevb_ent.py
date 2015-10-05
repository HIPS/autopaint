import autograd.numpy as np
from autopaint.util import entropy_of_a_diagonal_gaussian,entropy_of_diagonal_gaussians, neg_kl_diag_normal
from plotting import plot_density
from autopaint.aevb import compute_kl
import numpy.random as npr

def lower_bound(weights,encode,decode_log_like,log_prior,N_weights_enc,train_images,samples_per_image,latent_dimensions,rs):
    enc_w = weights[0:N_weights_enc]
    dec_w = weights[N_weights_enc:len(weights)]
    mean_log_joint,mean_ent = compute_log_prob_and_ent(enc_w,dec_w,encode,decode_log_like,log_prior,train_images,samples_per_image,latent_dimensions,rs)
    print "joint ll average",mean_log_joint
    print "ent average", mean_ent
    #
    # mean_log_prob = np.mean(compute_log_prob(enc_w,dec_w,encode,decode_log_like,train_images,samples_per_image,latent_dimensions,rs))
    # mean_kl = compute_kl(enc_w,train_images,encode)
    # print "ll average", mean_log_prob
    # print "neg kl average", mean_kl
    # print  mean_log_prob + mean_kl - (mean_log_joint + mean_ent)
    # assert  mean_log_prob + mean_kl == mean_log_joint + mean_ent
    return mean_log_joint + mean_ent

def compute_log_prob_and_ent(enc_w,dec_w,encode,decode_log_like,log_prior,train_images,samples_per_image,latent_dimensions,rs):
    (mus,log_sigs) = encode(enc_w,train_images)
    sigs = np.exp(log_sigs)
    noise = rs.randn(samples_per_image,train_images.shape[0],latent_dimensions)
    Z_samples = mus + sigs*noise
    Z_samples = np.reshape(Z_samples,(train_images.shape[0]*samples_per_image,latent_dimensions),order = 'F')
    mean_ent = np.mean(entropy_of_diagonal_gaussians(sigs))
    train_images_repeat = np.repeat(train_images,samples_per_image,axis=0)
    mean_log_prob = np.mean(decode_log_like(dec_w,Z_samples,train_images_repeat) +log_prior(Z_samples))
    # mean_kl = compute_kl(enc_w,train_images,encode)
    # print np.mean(log_prior(Z_samples)) + mean_ent - mean_kl
    # assert log_prior(Z_samples) + mean_ent == mean_kl
    return mean_log_prob, mean_ent



def enc_lower_bound(enc_w,dec_w,encode,decode_log_like,log_prior,N_weights_enc,train_images,samples_per_image,latent_dimensions,rs):
    #Like lowerbound, but we're assuming dec_w is given to us and will not be optimized
    mean_log_joint,mean_ent = compute_log_prob_and_ent(enc_w,dec_w,encode,decode_log_like,log_prior,train_images,samples_per_image,latent_dimensions,rs)
    print "joint ll average",mean_log_joint
    print "ent average", mean_ent
    return mean_log_joint + mean_ent


# def compute_log_prob(enc_w,dec_w,encode,decode_log_like,train_images,samples_per_image,latent_dimensions,rs):
#     (mus,log_sigs) = encode(enc_w,train_images)
#     sigs = np.exp(log_sigs)
#     noise = rs.randn(samples_per_image,train_images.shape[0],latent_dimensions)
#     Z_samples = mus + sigs*noise
#     Z_samples = np.reshape(Z_samples,(train_images.shape[0]*samples_per_image,latent_dimensions),order = 'F')
#     train_images_repeat = np.repeat(train_images,samples_per_image,axis=0)
#     mean_log_prob = decode_log_like(dec_w,Z_samples,train_images_repeat)
#     return mean_log_prob
#
# def compute_kl(enc_w,train_images,encode):
#     (mus,log_sigs) = encode(enc_w,train_images)
#     sigs = np.exp(log_sigs)
#     kl_vect = neg_kl_diag_normal(mus,sigs)
#     return np.mean(kl_vect)
#
# def cross_ent(mus,sigs):
#     return -1*.5*(np.log(2*np.pi)+sigs**2+mus**2)
#



