import autograd.numpy as np
from autopaint.util import neg_kl_diag_normal, entropy_of_diagonal_gaussians,build_logprob_standard_normal, entropy_of_a_diagonal_gaussian
from scipy.stats import entropy
def lower_bound(weights,encode,decode_log_like,N_weights_enc,base_data,conditional_data,samples_per_image,latent_dimensions,rs):
    enc_w = weights[0:N_weights_enc]
    dec_w = weights[N_weights_enc:len(weights)]
    mean_log_prob = np.mean(compute_log_prob(enc_w,dec_w,encode,decode_log_like,base_data,conditional_data,samples_per_image,latent_dimensions,rs))
    mean_kl = compute_kl(enc_w,conditional_data,encode)
    print "ll average", mean_log_prob
    print "neg kl average", mean_kl
    return mean_log_prob + mean_kl

def compute_log_prob(enc_w,dec_w,encode,decode_log_like,base_data,conditional_data,samples_per_image,latent_dimensions,rs):
    (mus,log_sigs) = encode(enc_w,conditional_data)
    sigs = np.exp(log_sigs)
    noise = rs.randn(samples_per_image,conditional_data.shape[0],latent_dimensions)
    Z_samples = mus + sigs*noise
    Z_samples = np.reshape(Z_samples,(conditional_data.shape[0]*samples_per_image,latent_dimensions),order = 'F')
    conditional_repeat = np.repeat(conditional_data,samples_per_image,axis=0)
    base_repeat = np.repeat(base_data,samples_per_image,axis=0)
    decoder_input = np.concatenate((Z_samples,base_repeat),axis = 1)
    mean_log_prob = decode_log_like(dec_w,decoder_input,conditional_repeat)
    return mean_log_prob

def compute_kl(enc_w,train_images,encode):
    (mus,log_sigs) = encode(enc_w,train_images)
    sigs = np.exp(log_sigs)
    kl_vect = neg_kl_diag_normal(mus,sigs)
    return np.mean(kl_vect)

def cross_ent(mus,sigs):
    return -1*.5*(np.log(2*np.pi)+sigs**2+mus**2)




