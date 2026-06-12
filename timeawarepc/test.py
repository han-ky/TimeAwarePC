from numpy.random import default_rng

rng = default_rng(seed=111)
import timeawarepc.tpc_mod as tpc_mod
import timeawarepc.tpc as tpc
from timeawarepc.simulate_data import simulate_data
import logging
import rpy2.robjects as ro

def main():
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format="%(asctime)s - %(levelname)s - %(message)s"
    # )
    ro.r('''source("renv/activate.R")''')
    model = 'nonlinnongauss'
    isGauss = model == 'lingauss'
    T = 1000
    noise = 1

    data, CFCtruth = simulate_data(model, T = T, noise = noise)

    print('Ground Truth CFC adjacency:')
    print(CFCtruth)

    alpha = 0.05
    maxdelay=1
    niter = 50
    thresh = 0.25

    adjmat, causaleffmat = tpc_mod.cfc_tpc(data,maxdelay=maxdelay,alpha=alpha,niter=niter,thresh=thresh)

    print('Estimated CFC adjacency by ball divergence CI test and dcor I test:')
    print(adjmat)

    adjmat, causaleffmat = tpc.cfc_tpc(data,maxdelay=maxdelay,alpha=alpha,niter=niter,thresh=thresh,isgauss=isGauss)

    print('Estimated CFC adjacency by original method(s):')
    print(adjmat)

if __name__ == "__main__":
    main()