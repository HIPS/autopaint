import autograd.numpy as np


def sga(value_and_grad,params,num_iters, alpha = .001,callback = None):
    # Simple stochastic gradient ascent
    D = len(params)
    for i in xrange(num_iters):
        val, grad = value_and_grad(params)
        if callback:
            callback(val,params)
        params = params + alpha * grad
    val = value_and_grad(params)
    return params,val

def adam(value_and_grad,params,num_iters, alpha = .001, beta1 = .9,beta2 = .999, eps = 1e-8,callback = None):
    # Maximizes a stochastic function using ADAM
    # http://arxiv.org/pdf/1412.6980v8.pdf
    D = len(params)
    m = np.zeros(D)
    v = np.zeros(D)
    for i in xrange(num_iters):
        val, grad = value_and_grad(params)
        if callback:
            callback(val,params)
        m = beta1*m + (1-beta1)*grad
        v = beta2*v + (1-beta2)*grad**2
        mhat = m/(1-beta1**(i+1))
        vhat = v/(1-beta2**(i+1))
        params = params + alpha * mhat/(np.sqrt(vhat)+eps)
    val = value_and_grad(params)
    return params,val

