import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 0, 1 # mean and standard deviation
data = np.random.normal(mu, sigma, 100)

plt.plot(data)
plt.show()