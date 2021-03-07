from gtda.diagrams import HeatKernel
import numpy as np


def heat_kernel(dgm, n_jobs=1, width=128, sigma=0.1):
    kh = HeatKernel(sigma=sigma, n_bins=width, n_jobs=n_jobs)

    cfst = kh.fit_transform([dgm])[0]
    return cfst
    # return  np.round(np.transpose(cfst, (1,2,0))*255).astype(np.int32)
