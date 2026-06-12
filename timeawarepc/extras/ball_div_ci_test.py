import numpy as np
from scipy.stats import iqr
from scipy.spatial.distance import cdist
from dcor.independence import distance_covariance_test
from timeawarepc import simulate_data

def i_test_cond_ball_div_uncond_dist_cov(data, A, B, S, niter=50, **kwargs):
    if len(S) == 0:
        return distance_covariance_test(data[:, A], data[:, B], num_resamples=niter).pvalue
    return ci_test_ball_div(data, A, B, S, niter=niter)

def ci_test_ball_div(data, A, B, S, niter=500):
    """
    Conditional independence test based on the conditional ball divergence (cBD),
    calibrated via the local wild bootstrap.
 
    Reference: Banerjee, Bhattacharya & Ghosh (2024),
    "A Ball Divergence Based Measure For Conditional Independence Testing."
 
    Parameters
    ----------
    data : np.ndarray, shape (n, p)
        Data matrix; rows are observations, columns are variables.
    A : int
        Column index of X (the tested variable).
    B : int
        Column index of Y (the other tested variable).
    S : set of int
        Column index/indices of Z (the conditioning set).
 
    Returns
    -------
    pval : float
        Bootstrap p-value for H0: X _||_ Y | Z.
    """

    n = data.shape[0]

    dX = 1
    dY = 1
    dZ = len(S)

    X = data[:, A].reshape(-1, dX)
    Y = data[:, B].reshape(-1, dY)
    Z = data[:, sorted(S)]

    iqrZ = iqr(Z, axis=0) #choice described in section 5 of Banerjee  
    iqrY = iqr(Y)
    c1 = np.mean(np.append(iqrZ, iqrY))
    c2 = np.mean(iqrZ)

    h1 = c1 * n ** (-1/(dY + dZ + 2)) #choice described in section 5 of Banerjee  
    h2 = c2 * n ** (-1/(dZ + 2)) 
    h0 = 20 * c2 * n ** (-1 / 1.95)

    def epank(x):
        return 0.75 * (1.0 - x ** 2) * (np.abs(x) <= 1.0)
    
    YZ = np.hstack((Y, Z))
    distYZ = cdist(YZ, YZ)
    distZ = cdist(Z, Z)

    wyz = epank(distYZ / h1)
    wz = epank(distZ / h2)
    colsum_wyz = np.maximum(wyz.sum(axis=0), 1e-300) #guard against div by zero
    colsum_wz = np.maximum(wz.sum(axis=0), 1e-300) #guard against div by zero
    wyz_norm = wyz / colsum_wyz
    wz_norm = wz / colsum_wz

    wdiff = wyz_norm - wz_norm
    wyz_outer = np.expand_dims(wyz_norm, axis=1) * np.expand_dims(wyz_norm, axis=0)
    wz_outer = np.expand_dims(wz_norm, axis=1) * np.expand_dims(wz_norm, axis=0)
    w_outer = wyz_outer + wz_outer

    def calc_zeta_hat(X_vals):
        distX = cdist(X_vals, X_vals)
        delta = np.expand_dims(distX, axis=1) <= np.expand_dims(distX, axis=-1)
        inner = np.matmul(delta.reshape(-1, n), wdiff).reshape(n, n, n) # same as np.matmul(delta, wdiff), reshape first to speed up matmul
        result = np.einsum('uvs,uvs,uvs->', inner, inner, w_outer) / n # take weight function a(.,.) = 1 constant per D.2 of Banerjee paper supplementary material
        return result

    zeta_hat = calc_zeta_hat(X)

    zeta_bootstrap = np.empty(niter)
    rng = np.random.default_rng()
    
    for i in range(niter):
       X_perturb = rng.normal(loc=X, scale=h0)
       zeta_bootstrap[i] = calc_zeta_hat(X_perturb)

    pval = (1.0 + np.sum(zeta_bootstrap >= zeta_hat)) / (1.0 + niter)

    return pval

def main():
    print('Testing ball divergence CI test...')
    model = 'lingauss'
    T = 100
    noise = 1
    data, CFCtruth = simulate_data.simulate_data(model, T, noise)
    test = i_test_cond_ball_div_uncond_dist_cov(data, 0, 2, {1,3}, niter=1000)
    print(test)

if __name__ == "__main__":
    main()