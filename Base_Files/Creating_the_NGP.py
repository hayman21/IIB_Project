import numpy as np
from Base_Files.Kernel_Functions import GaussianKernel


def GaussianProcess(coords, l, N):
    K = GaussianKernel(coords, l)

    # Generate a vector of Gaussian vales - N(0, I)
    z = np.random.randn(N)

    # Calculate the Cholesky decomposition of K. This is the matric such that K = L L^T
    L = np.linalg.cholesky(K + 1e-6 * np.eye(N))

    # Matrix multiply L and z
    u = L @ z

    return u
