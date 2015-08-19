import autograd.numpy as np
import time

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

def sga_mini_batch(batch_value_and_grad,params,batch_idxs,num_epochs, alpha = .001,callback = None):
    # Simple stochastic gradient ascent
    D = len(params)
    for i in xrange(num_epochs):
        for idxs in batch_idxs:
            start = time.time()
            val, grad = batch_value_and_grad(params,idxs)
            if callback:
                callback(val,params)
            params = params + alpha * grad
            end = time.time()
            print 'time per iter:', end - start
    val, grad = batch_value_and_grad(params,idxs)
    return params,val

def adam(value_and_grad,params,num_iters, alpha = .001, beta1 = .9,beta2 = .999, eps = 1e-8,callback = None):
    # Maximizes a stochastic function using ADAM
    # http://arxiv.org/pdf/1412.6980v8.pdf
    D = len(params)
    m = np.zeros(D)
    v = np.zeros(D)
    for i in xrange(num_iters):
        start = time.time()
        val, grad = value_and_grad(params)
        if callback:
            callback(val,params)
        m = beta1*m + (1-beta1)*grad
        v = beta2*v + (1-beta2)*grad**2
        mhat = m/(1-beta1**(i+1))
        vhat = v/(1-beta2**(i+1))
        params = params + alpha * mhat/(np.sqrt(vhat)+eps)
        end = time.time()
        print 'time per iter:', end - start
    val = value_and_grad(params)
    return params,val


def adam_mini_batch(batch_value_and_grad,params,batch_idxs,num_epochs, alpha = .001, beta1 = .9,beta2 = .999, eps = 1e-8,callback = None):
    # Maximizes a stochastic function using ADAM in minibatch
    # http://arxiv.org/pdf/1412.6980v8.pdf
    # batch_value_and_grad is a function that returns the value and gradient of the function at params at idxs in batch_idx
    D = len(params)
    m = np.zeros(D)
    v = np.zeros(D)
    for i in xrange(num_epochs):
        for idxs in batch_idxs:
            start = time.time()
            val, grad = batch_value_and_grad(params,idxs)
            if callback:
                callback(val,params)
            m = beta1*m + (1-beta1)*grad
            v = beta2*v + (1-beta2)*grad**2
            mhat = m/(1-beta1**(i+1))
            vhat = v/(1-beta2**(i+1))
            params = params + alpha * mhat/(np.sqrt(vhat)+eps)
            end = time.time()
            print 'time per iter:', end - start
    val, grad = batch_value_and_grad(params,idxs)
    return params,val

def ada_grad_mini_batch(batch_value_and_grad,params,batch_idxs,num_epochs, alpha = 1, eps = 1e-8,callback = None):
    # Maximizes a stochastic function using adagrad in minibatch
    # batch_value_and_grad is a function that returns the value and gradient of the function at params at idxs in batch_idx
    D = len(params)
    grad_sq_hist = np.zeros(D)
    for i in xrange(num_epochs):
        for idxs in batch_idxs:
            start = time.time()
            val, grad = batch_value_and_grad(params,idxs)
            if callback:
                callback(val,params)
            grad_sq_hist = grad_sq_hist + grad**2
            params = params + alpha*1/(np.sqrt(grad_sq_hist)+eps)*grad
            end = time.time()
            print 'time per iter:', end - start
    val, grad = batch_value_and_grad(params,idxs)
    return params,val


def ada_delta(value_and_grad,params,num_iters, rho = .001, eps = 1e-8,callback = None):
    # Maximizes a stochastic function using ADADELTA
    # http://arxiv.org/pdf/1212.5701v1.pdf
    D = len(params)
    g_sq = np.zeros(D)
    d_sq = np.zeros(D)
    for i in xrange(num_iters):
        val, grad = value_and_grad(params)
        if callback:
            callback(val,params)
        g_sq = rho*g_sq + (1-rho)*grad**2
        d = np.sqrt(d_sq+eps)/np.sqrt(g_sq+eps)*grad
        d_sq = rho*d_sq + (1-rho)*d**2
        params = params+d
    val = value_and_grad(params)
    return params,val
