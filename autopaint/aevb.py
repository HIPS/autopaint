import autograd.numpy as np
from autopaint.util import neg_kl_diag_normal

def lower_bound(weights,encode,decode_log_like,N_weights_enc,train_images,samples_per_image,latent_dimensions,rs):
    enc_w = weights[0:N_weights_enc]
    dec_w = weights[N_weights_enc:len(weights)]
    mean_log_prob = np.mean(compute_log_prob(enc_w,dec_w,encode,decode_log_like,train_images,samples_per_image,latent_dimensions,rs))
    mean_kl = compute_kl(enc_w,train_images,encode)
    print "ll average", mean_log_prob
    print "kl average", mean_kl
    return mean_log_prob + mean_kl

def compute_log_prob(enc_w,dec_w,encode,decode_log_like,train_images,samples_per_image,latent_dimensions,rs):
    (mus,log_sigs) = encode(enc_w,train_images)
    sigs = np.exp(log_sigs)
    noise = rs.randn(samples_per_image,train_images.shape[0],latent_dimensions)
    Z_samples = mus + sigs*noise
    Z_samples = np.reshape(Z_samples,(train_images.shape[0]*samples_per_image,latent_dimensions),order = 'F')
    train_images_repeat = np.repeat(train_images,samples_per_image,axis=0)
    mean_log_prob = decode_log_like(dec_w,Z_samples,train_images_repeat)
    return mean_log_prob

def compute_kl(enc_w,train_images,encode):
    (mus,log_sigs) = encode(enc_w,train_images)
    sigs = np.exp(log_sigs)
    kl_vect = neg_kl_diag_normal(mus,sigs)
    return np.mean(kl_vect)


