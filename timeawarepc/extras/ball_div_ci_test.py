def ci_test_ball_div(data,A,B,S,**kwargs):
    from scipy.stats import iqr
    import numpy as np
    from scipy.spatial.distance import cdist

    niter = 500

    n = data.shape[0]
    p = data.shape[1]

    idx = np.zeros(p, dtype=bool)
    for i in range(p):
        if i in S:
            idx[i] = True

    X = data[:,A]
    Y = data[:,B]
    Z = data[:,idx]

    dX = 1
    dY = 1
    dZ = np.sum(idx)

    iqrZ = iqr(Z, axis = 0) #choice described in section 5 of Banerjee  
    iqrY = iqr(Y)
    c1 = np.mean(np.append(iqrZ, iqrY))
    c2 = np.mean(iqrZ)

    h1 = c1 * n ** (-1/(dY + dZ + 2)) #choice described in section 5 of Banerjee  
    h2 = c2 * n ** (-1/(dZ + 2)) 
    h0 = 20 * c2 * n ** (-1 / 1.95)

    def epank(x):
        return 0.75 * (1 - x ** 2) * (np.abs(x) <= 1)
    
    YZ = np.hstack((Y.reshape(-1,1), Z))
    distYZ = cdist(YZ, YZ, 'euclidean')
    distZ = cdist(Z, Z, 'euclidean')

    wyz = epank(distYZ / h1)
    wz = epank(distZ / h2)
    wyz_sum = np.sum(wyz, axis = 0)
    wz_sum = np.sum(wz, axis = 0)
    wyz_norm = wyz / wyz_sum
    wz_norm = wz / wz_sum

    def calc_zeta_hat(X, wyz_norm, wz_norm):
        
        return result
    
    zeta_hat = calc_zeta_hat(X, wyz_norm, wz_norm)

    zeta_bootstrap = np.zeros(niter)
    for i in range(niter):
        rng = np.random.default_rng()
        Xhat = rng.normal(loc = X, scale = h0**2)
        zeta_bootstrap[i] = calc_zeta_hat(Xhat, wyz_norm, wz_norm)

    pval = (1 + np.sum(zeta_bootstrap >= zeta_hat)) / (1 + niter)

    return pval