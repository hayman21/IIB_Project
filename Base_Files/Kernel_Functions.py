import numpy as np
from scipy.spatial.distance import pdist, squareform

def GaussianKernel(coords, l):
    D = squareform(pdist(coords))
    return np.exp(-D**2/(2*l**2))
