import numpy as np
import matplotlib.pyplot as plt
from Base_Files.Creating_the_NGP import GaussianProcess


N = 1000
coords = np.linspace(0, 10, N).reshape(-1,1)
l = 1

plt.figure().set_figwidth(12)
plt.plot(coords, GaussianProcess(coords, l, N))
plt.show()