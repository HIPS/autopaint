import autograd.numpy as np

def sga(gradfun, params, num_iters, alpha=.001, callback=None):
    # Stochastic gradient ascent
    for i in xrange(num_iters):
        grad = gradfun(params, i)
        if callback:
            callback(params, i, grad)
        params = params + alpha * grad
    return params

def sga_momentum(grad, x, num_iters=200, step_size=0.1, mass=0.9, callback=None):
    """Stochastic gradient ascent with momentum.
    grad() must have signature grad(x, i), where i is the iteration number."""
    velocity = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x)
        if callback: callback(x, i, g)
        velocity = mass * velocity - (1.0 - mass) * g
        x = x + step_size * velocity
    return x

def adam(gradfun, params, num_iters, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8, callback=None):
    # Maximizes a stochastic function using Adam.
    # http://arxiv.org/pdf/1412.6980v8.pdf
    D = len(params)
    m = np.zeros(D)
    v = np.zeros(D)
    for i in xrange(num_iters):
        grad = gradfun(params, i)
        if callback:
            callback(params, i, grad)
        m = beta1*m + (1-beta1)*grad
        v = beta2*v + (1-beta2)*grad**2
        mhat = m/(1-beta1**(i+1))
        vhat = v/(1-beta2**(i+1))
        params = params + alpha * mhat/(np.sqrt(vhat)+eps)
    return params

def adagrad(gradfun, params, num_iters, alpha=1.0, eps=1e-8, callback=None):
    # Maximizes a stochastic function using adagrad.
    D = len(params)
    grad_sq_hist = np.zeros(D)
    for i in xrange(num_iters):
        val, grad = gradfun(params, i)
        if callback:
            callback(params, i, grad)
        grad_sq_hist = grad_sq_hist + grad**2
        params = params + alpha * 1.0/(np.sqrt(grad_sq_hist) + eps) * grad
    return params

def adadelta(gradfun, params, num_iters, rho=0.001, eps=1e-8, callback=None):
    # http://arxiv.org/pdf/1212.5701v1.pdf
    D = len(params)
    g_sq = np.zeros(D)
    d_sq = np.zeros(D)
    for i in xrange(num_iters):
        grad = gradfun(params, i)
        if callback:
            callback(params, i, grad)
        g_sq = rho*g_sq + (1.0 - rho)*grad**2.0
        d = np.sqrt(d_sq + eps)/np.sqrt(g_sq + eps)*grad
        d_sq = rho*d_sq + (1.0 - rho)*d**2.0
        params = params + d
    return params
